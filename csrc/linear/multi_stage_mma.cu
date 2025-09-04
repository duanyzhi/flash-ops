#include <cuda_fp16.h>
#include <mma.h>
#include "gemm.h"
#include "cuda_utils.cuh"

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

namespace flash_ops {

#define MAX(a, b) (a) > (b) ? (a) : (b)

template <int BM, int BN, int BK>
__device__ void load_smema(half* smema, const half* __restrict__ a, int K, int ik) {
  // load a: [128, 32] [M, K], row major
  // tid = 0 load 8 ele, row = 0, col = 0...7
  // tid = 1 load 8 ele, row = 0, col = 8...15
  // tid = 2 load 8 ele, row = 0, col = 16...23
  // tid = 3 load 8 ele, row = 0, col = 24...31
  // tid = 4 load 8 ele, row = 1, col = 0...7
  // ....
  // ik is the loop for K dim, step is BK
  int tid = threadIdx.x + threadIdx.y * 32 + threadIdx.z * 2 * 32; // thread num = 128

  for (int i = 0; i < 4; ++i) {
    int row = i * 32 + tid / 4;
    int col = tid % 4 * 8;
  
    void* ptr = (void*)(smema + row * BK + col);
    uint32_t smem_ptr; 
    asm(
      "{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }\n"
      : "=r"(smem_ptr) : "l"(ptr)
    );
    asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" :: "r"(smem_ptr),
                 "l"(&a[row * K + col + ik * BK]), "n"(16));
  }
}

template <int BM, int BN, int BK>
__device__ void load_smemb(half* smemb, const half* __restrict__ b, int K, int ik) {
  // load b: [64, 32] [N, K], colmajor
  int tid = threadIdx.x + threadIdx.y * 32 + threadIdx.z * 2 * 32;
  
  for (int i = 0; i < 2; ++i) {
    int col = i * 32 + tid / 4;
    int row = tid % 4 * 8;
  
    void* ptr = (void*)(smemb + col * BK + row);
    uint32_t smem_ptr; 
    asm(
      "{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }\n"
      : "=r"(smem_ptr) : "l"(ptr)
    );
    asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" :: "r"(smem_ptr),
                 "l"(&b[col * K + row + ik * BK]), "n"(16));
  }
}

template <int BM, int BN, int BK, int WMMA_M, int WMMA_N, int WMMA_K>
__device__ void load_fraga(half* smema, nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major>* frag_a, int ik){
  // load 4 fraga use warp0 and warp1
  // warp 0 -> a ik = 0 -> (0, 0) && ik = 1 -> (0, 1)
  // warp 1 -> a ik = 0 -> (1, 0) && ik = 1 -> (1, 1)
  const int row = threadIdx.y;
  // const int col = threadIdx.z;

  half* smemea_ptr = smema + row * 64 * (2 * 16) + ik * 16;
  for (int i = 0; i < 4; ++i) {
    int frag_row = i * WMMA_M;
    nvcuda::wmma::load_matrix_sync(frag_a[i], smemea_ptr + frag_row * (2 * 16), BK);
  }
}

template <int BM, int BN, int BK, int WMMA_M, int WMMA_N, int WMMA_K>
__device__ void load_fragb(half* smemb, nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::col_major>* frag_b, int ik){
  // tidy = 0, tidz = 0 ->
  // const int row = threadIdx.y;
  const int col = threadIdx.z;
  int tid = threadIdx.x + threadIdx.y * 32 + threadIdx.z * 2 *32;
  // printf("debug load fragb : tid %d col %d", tid, col);

  half* smemeb_ptr = smemb + ik * 16 + col * 32 * (2 * 16); // WARP_N = 32
  for (int i = 0; i < 2; ++i) {
    int frag_col = i * WMMA_N;
    nvcuda::wmma::load_matrix_sync(frag_b[i], smemeb_ptr + frag_col * (2 * 16), BK);
  }
}

template <int BM, int BN, int BK, int WMMA_M, int WMMA_N, int WMMA_K>
__device__ void store_fragc(float* smemc, nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>* frag_c, int frag_a_num, int frag_b_num){
  const int row = threadIdx.y;
  const int col = threadIdx.z;
  float* smemec_ptr = smemc + row * WMMA_M * frag_a_num * BN + col * WMMA_N * frag_b_num;
#pragma unroll
  for (int r = 0; r < frag_a_num; ++r) {
#pragma unroll
    for (int c = 0; c < frag_b_num; ++c) {
      nvcuda::wmma::store_matrix_sync(smemec_ptr + r * WMMA_M * BN + c * WMMA_N, frag_c[r * frag_b_num + c], BN, nvcuda::wmma::mem_row_major);
    }
  }
}

template <int BM, int BN, int BK>
__device__ void store2globalc(float* smemc, float* gmemc, int bx, int by, int N){
  /***
   *  --> bx
   *  |
   *  V
   *  by
   * ***/
  // one thread load 128 num, thread num = 128
  const int row = threadIdx.x + threadIdx.y * 32 + threadIdx.z * 2 * 32;
  for (int col = 0; col < BN; ++col) {
      gmemc[(by * BM + row) * N + bx * BN + col] = smemc[row * BN + col];
      // gmemc[(by * BM + row) * N + bx * BN + col] = __float2half(smemc[row * BN + col]);
  }
}

template <int WMMA_M, int WMMA_N, int WMMA_K>
__device__ void print_fraga(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major>& frag, half* data) {
  const int tid = threadIdx.x + threadIdx.y * 32 + threadIdx.z * 32 * 2;
  int lane_id = tid % 32;

  // __shared__ half smem[smem_num];
  // half* s_a = smem;
  for (int i = 0; i < frag.num_elements; i++) {
    int row{0}, col{0};
    int groupID = lane_id >> 2;
    int threadID_in_group = lane_id % 4;
    if (i < 4) {
      row = i < 2 ? groupID : groupID + 8;
      col = (threadID_in_group * 2) + (i & 0x1);
    } else {
      int j = i - 4;
      row = j < 2 ? groupID : groupID + 8;
      col = (threadID_in_group * 2) + (j & 0x1) + 8;
    }
    // printf("tid = %d row col = (%d, %d), v = %f \n", tid, row, col, __half2float(frag.x[i]));

    // int row_smem = m_index * WMMA_M + row;
    // int col_smem = n_index * WMMA_N + col;

    data[row * 16 + col] = frag.x[i];
  }
}

template <int WMMA_M, int WMMA_N, int WMMA_K>
__device__ void print_fragb(nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::col_major>& frag, half* data) {
  const int tid = threadIdx.x + threadIdx.y * 32 + threadIdx.z * 32 * 2;
  int lane_id = tid % 32;

  // __shared__ half smem[smem_num];
  // half* s_a = smem;
  for (int i = 0; i < frag.num_elements; i++) {
    int row{0}, col{0};
    int groupID = lane_id >> 2;
    int threadID_in_group = lane_id % 4;
    if (i < 4) {
      if (i < 2) {
        row = (threadID_in_group * 2) + (i & 0x1);
      } else {
        row = (threadID_in_group * 2) + (i & 0x1) + 8;
      }
      col = groupID;
    } else {
      if (i - 4 < 2) {
        row = (threadID_in_group * 2) + (i & 0x1);
      } else {
        row = (threadID_in_group * 2) + (i & 0x1) + 8;
      }
      col = groupID + 8; 
    }
    

    // if (i < 4) {
    //   row = i < 2 ? groupID : groupID + 8;
    //   col = (threadID_in_group * 2) + (i & 0x1);
    // } else {
    //   int j = i - 4;
    //   row = j < 2 ? groupID : groupID + 8;
    //   col = (threadID_in_group * 2) + (j & 0x1) + 8;
    // }
    // printf("tid = %d row col = (%d, %d), v = %f \n", tid, row, col, __half2float(frag.x[i]));

    // int row_smem = m_index * WMMA_M + row;
    // int col_smem = n_index * WMMA_N + col;

    data[col * 16 + row] = frag.x[i];

    // smem_float[row_smem * BN + col_smem] = frag_c[m_index][n_index].x[i];
  }
}


template <int WMMA_M, int WMMA_N, int WMMA_K>
__device__ void print_fragc(nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>& frag, float* data) {
  const int tid = threadIdx.x + threadIdx.y * 32 + threadIdx.z * 32 * 2;
  int lane_id = tid % 32;

  // __shared__ half smem[smem_num];
  // half* s_a = smem;
  for (int i = 0; i < frag.num_elements; i++) {
    int row{0}, col{0};
    int groupID = lane_id >> 2;
    int threadID_in_group = lane_id % 4;
    if (i < 4) {
      row = i < 2 ? groupID : groupID + 8;
      col = (threadID_in_group * 2) + (i & 0x1);
    } else {
      int j = i - 4;
      row = j < 2 ? groupID : groupID + 8;
      col = (threadID_in_group * 2) + (j & 0x1) + 8;
    }
    // printf("tid = %d row col = (%d, %d), v = %f \n", tid, row, col, frag.x[i]);

    // int row_smem = m_index * WMMA_M + row;
    // int col_smem = n_index * WMMA_N + col;

    data[row * 16 + col] = frag.x[i];
  }
}

template <int BM, int BN, int BK, int WARP_M, int WARP_N, int WARP_K, int WMMA_M, int WMMA_N, int WMMA_K, int THREADNUM, int STAGE>
__global__ void multi_stage_mma_half_kernel(
    const half* __restrict__ a, const half* __restrict__ b,
    const half* __restrict__ bias, float* __restrict__ c, const int M,
    const int N, const int K) {
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  const int tid_y = threadIdx.y;
  const int tid_z = threadIdx.z;

  const int tid = threadIdx.x + tid_y * 32 + tid_z * 32 * 2;

  constexpr int warp_size = 32;
  // int wid = tid / warp_size;
  // int lane_id = tid % warp_size;

  constexpr int warp_num = 4;

  int load_block_a_gmem_addr = by * BM * K;
  int load_block_b_gmem_addr = bx * BN * K;

  constexpr int smem_size = (BM * BK + BN * BK) * STAGE;

  // static_assert(smem_size >= (48 << 10) && "smem_num must be small 49152");

  __shared__ half smem[smem_size];
  half* smema = smem;
  half* smemb = smem + BM * BK;
  half* smema2 = smemb + BN * BK;
  half* smemb2 = smema2 + BM * BK;
  float* smemc = reinterpret_cast<float *>(smem); // 32768

  // prologue
  load_smema<BM, BN, BK>(smema, a + load_block_a_gmem_addr, K, 0);
  load_smemb<BM, BN, BK>(smemb, b + load_block_b_gmem_addr, K, 0);

  asm volatile("cp.async.commit_group;\n" ::);

  load_smema<BM, BN, BK>(smema2, a + load_block_a_gmem_addr, K, 1);
  load_smemb<BM, BN, BK>(smemb2, b + load_block_b_gmem_addr, K, 1);

  asm volatile("cp.async.commit_group;\n" ::);

  // __syncthreads(); 
  // if (tid == 0) {
  //  for (int c = 0; c < 4; ++c) {
  //   for (int i = 0; i < 32; ++i) {
  //     printf("%f ", __half2float(smemb[i + c * 32]));
  //   }
  //   printf("\n");
  //  }
  // }
  // // print_smem(smema, BM, BK);

  constexpr int frag_a_num = WARP_M / WMMA_M;  // 64 / 16 = 4
  constexpr int frag_b_num = WARP_N / WMMA_N;  // 32 / 16 = 2
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> frag_a[frag_a_num];
  // fragement_a_type frag_a[frag_a_num];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::col_major> frag_b[frag_b_num];
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_c[frag_a_num * frag_b_num];
  // nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_cc;

  for (int m_index = 0; m_index < frag_a_num; m_index++) {
    for (int n_index = 0; n_index < frag_b_num; n_index++) {
      nvcuda::wmma::fill_fragment(frag_c[m_index * frag_b_num + n_index], 0.0f);
    }
  }
  // nvcuda::wmma::fill_fragment(frag_cc, 0.0f);
  /***
   * 
   *    -----------------------
   *    |  warp_0   |  warp_1  |
   *    |  warp_2   |  warp_3  |
   *    -----------------------
   * 
   * ***/
#pragma unroll
  for (int lk = 0; lk < K / BK - 2; lk = lk + 2) {
    // if (tid == 0) {
    //   printf("lk = %d \n", lk);
    // }
  // for (int lk = 0; lk < 1; ++lk) {
    // for loop for mma
    // one wrap 64x64x16, one mma 16x16x16
    asm volatile("cp.async.wait_group %0;\n" ::"n"(1));
    __syncthreads();
#pragma unroll
    for (int ik = 0; ik < BK / WARP_K; ik += 1) {
      load_fraga<BM, BN, BK, WMMA_M, WMMA_N, WMMA_K>(smema, frag_a, ik);
      load_fragb<BM, BN, BK, WMMA_M, WMMA_N, WMMA_K>(smemb, frag_b, ik);

#pragma unroll
      for (int m_index = 0; m_index < frag_a_num; m_index++) {
#pragma unroll
        for (int n_index = 0; n_index < frag_b_num; n_index++) {
          // do mma and sum all ik
          // if (m_index == 0 && n_index == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
            nvcuda::wmma::mma_sync(frag_c[m_index * frag_b_num + n_index], frag_a[m_index], frag_b[n_index],
                                   frag_c[m_index * frag_b_num + n_index]);
        }
      }
    }

    asm volatile("cp.async.wait_group %0;\n" ::"n"(1));
    __syncthreads();


    // do lk + 2 load
    load_smema<BM, BN, BK>(smema, a + load_block_a_gmem_addr, K, lk + 2);
    load_smemb<BM, BN, BK>(smemb, b + load_block_b_gmem_addr, K, lk + 2);
    asm volatile("cp.async.commit_group;\n" ::);

    // stage 2
    #pragma unroll
    for (int ik = 0; ik < BK / WARP_K; ik += 1) {
      load_fraga<BM, BN, BK, WMMA_M, WMMA_N, WMMA_K>(smema2, frag_a, ik);
      load_fragb<BM, BN, BK, WMMA_M, WMMA_N, WMMA_K>(smemb2, frag_b, ik);

#pragma unroll
      for (int m_index = 0; m_index < frag_a_num; m_index++) {
#pragma unroll
        for (int n_index = 0; n_index < frag_b_num; n_index++) {
          // do mma and sum all ik
          // if (m_index == 0 && n_index == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
          nvcuda::wmma::mma_sync(frag_c[m_index * frag_b_num + n_index], frag_a[m_index], frag_b[n_index],
                                 frag_c[m_index * frag_b_num + n_index]);
        }
      }
    }

    // do lk + 3 load
    load_smema<BM, BN, BK>(smema2, a + load_block_a_gmem_addr, K, lk + 3);
    load_smemb<BM, BN, BK>(smemb2, b + load_block_b_gmem_addr, K, lk + 3);
    asm volatile("cp.async.commit_group;\n" ::);

  }
  {
    // last 2 mma
    asm volatile("cp.async.wait_group %0;\n" ::"n"(1));
    __syncthreads();
#pragma unroll
    for (int ik = 0; ik < BK / WARP_K; ik += 1) {
      load_fraga<BM, BN, BK, WMMA_M, WMMA_N, WMMA_K>(smema, frag_a, ik);
      load_fragb<BM, BN, BK, WMMA_M, WMMA_N, WMMA_K>(smemb, frag_b, ik);

#pragma unroll
      for (int m_index = 0; m_index < frag_a_num; m_index++) {
#pragma unroll
        for (int n_index = 0; n_index < frag_b_num; n_index++) {
          // do mma and sum all ik
          // if (m_index == 0 && n_index == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
          nvcuda::wmma::mma_sync(frag_c[m_index * frag_b_num + n_index], frag_a[m_index], frag_b[n_index],
                                 frag_c[m_index * frag_b_num + n_index]);
        }
      }
    }

    // last one
    asm volatile("cp.async.wait_group %0;\n" ::"n"(1));
    __syncthreads();
#pragma unroll
    for (int ik = 0; ik < BK / WARP_K; ik += 1) {
      load_fraga<BM, BN, BK, WMMA_M, WMMA_N, WMMA_K>(smema2, frag_a, ik);
      load_fragb<BM, BN, BK, WMMA_M, WMMA_N, WMMA_K>(smemb2, frag_b, ik);

#pragma unroll
      for (int m_index = 0; m_index < frag_a_num; m_index++) {
#pragma unroll
        for (int n_index = 0; n_index < frag_b_num; n_index++) {
          // do mma and sum all ik
          // if (m_index == 0 && n_index == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
          nvcuda::wmma::mma_sync(frag_c[m_index * frag_b_num + n_index], frag_a[m_index], frag_b[n_index],
                                 frag_c[m_index * frag_b_num + n_index]);
        }
      }
    }
  }

  // store 
  store_fragc<BM, BN, BK, WMMA_M, WMMA_N, WMMA_K>(smemc, frag_c, frag_a_num, frag_b_num);
  __syncthreads();
  store2globalc<BM, BN, BK>(smemc, c, bx, by, N);
} 

at::Tensor multi_stage_mma_forward(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias) {
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    const auto input_dim = input.dim();
    int64_t M = -1;
    const int64_t N = weight.size(0);
    const int64_t K = weight.size(1);
    std::vector<int64_t> output_size;
    if (input_dim == 3) {
        TORCH_CHECK(input.size(2) == weight.size(1),
            "Input and weight dimensions mismatch");
        M = input.size(0) * input.size(1);
        output_size = {input.size(0), input.size(1), N};
    } else if (input_dim == 2) {
        TORCH_CHECK(input.size(1) == weight.size(1),
            "Input and weight dimensions mismatch");
        M = input.size(0);
        output_size = {M, N};
    } else {
        AT_ERROR(
        "Expected gemm input dim be 2 or 3, but got ", input_dim, "."
        );
    }

    at::Tensor output = at::zeros(
        output_size, 
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)
        // torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA)
    );
    // at::Tensor c = at::zeros(output_size, input.options());

    // block tile (BM, BN, BK) = (128, 128, 32)
    // warp tile (WARP_M, WARP_N, WARP_K) = (64, 64, 16)
    // mma tile (WMMA_M, WMMA_N, WMMA_K) = (16, 16, 16)
    // one warp process 64 * 64 * 16, so need 4 warp
    // one warp need 64 / 16 * 64 / 16 * 16 / 16 = 16 time wmma

    constexpr int BM = 128;
    constexpr int BN = 64;
    constexpr int BK = 32;
    // A &&B (128 * 32 + 64 * 32） * 2（half) * 3(stage) = (4096 + 2048) * 6 = 36864
    // shared C: 128 * 64 * 4 = 32768

    constexpr int WARP_M = 64;
    constexpr int WARP_N = 32;
    constexpr int WARP_K = 16;

    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;
    constexpr int thread_num = 32 * 4; // 4 warp
    constexpr int stage = 3; // 3 pingpong pipeline

    dim3 GridDim(DIV_UP(N, BN), DIV_UP(M, BM));
    dim3 BlockDim(32, 2, 2);
    // float* c_ptr = reinterpret_cast<float*>(output.data_ptr());

    // using fragement_a_type = nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major>;
    multi_stage_mma_half_kernel<BM, BN, BK, WARP_M, WARP_N, WARP_K, WMMA_M, WMMA_N, WMMA_K, thread_num, stage>
        <<<GridDim, BlockDim, 0, stream>>>(
               reinterpret_cast<__half*>(input.data_ptr<at::Half>()),
               reinterpret_cast<__half*>(weight.data_ptr<at::Half>()),
               nullptr,
               reinterpret_cast<float*>(output.data_ptr()),
                // reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
               M, N, K);

    cudaStreamSynchronize(stream);
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
    return output;
}


} // flash_ops
