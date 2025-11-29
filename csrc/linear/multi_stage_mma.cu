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
    
    col = col ^(((row &3)<<3));
  
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
  
  for (int i = 0; i < 4; ++i) {
    int col = i * 32 + tid / 4;
    int row = tid % 4 * 8;

    col = col ^ (((row & 3) << 3));
  
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
__device__ void load_fraga(half *smem, unsigned int *frag, int ik){
  // T0: (0,0),(0,1),(8,0),(8,1),(0,8),(0,9),(8,8),(8,9)
  // T1: (1,0),(1,1),(9,0),(9,1),(1,8),(1,9),(9,8),(9,9)
  // T2: (2,0),(2,1),(10,0),(10,1),(2,8),(2,9),(10,8),(10,9)
  //　...
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  const int tid = tx + ty * 32;
  // const int smem_row = threadIdx.y;
  // const int semm_col = ik;
  // half* smemea_ptr = smema + smem_row * 64 * (2 * 16) + ik * 16;

  int lane_id = tid % 32;
  for (int n = 0; n < 4; ++n) {  // n for 4 fragement
    int row_start = threadIdx.y * 64 + n * 16;
    int col_start = ik * 16;
    for (int i = 0; i < 8; i = i + 2) {  // i = 0 2 4 6
      int row{0}, col{0};
      int groupID = lane_id >> 2;
      int threadID_in_group = lane_id % 4;
      
      if (i < 2 || (i >=4 && i < 6)) {
        row = groupID;
      } else {
        row = groupID + 8;
      }
      if (i < 4) {
        col = (threadID_in_group * 2) + (i & 0x1);
      } else {
        col = (threadID_in_group * 2) + (i & 0x1) + 8;
      }

      row = row_start + row;
      col = col_start + col;
      
      frag[n * 4 + i/2] = *(reinterpret_cast<unsigned int *>(smem + row * 32 + col));
    }
  }
}

template <int BM, int BN, int BK, int WMMA_M, int WMMA_N, int WMMA_K>
__device__ void load_fragb(half* smem, unsigned int *frag, int ik){
  int tx = threadIdx.x;
  int tz = threadIdx.z;
  
  const int tid = tx + 2 * 32 + tz * 32 * 2;
  int lane_id = tid % 32;

  for (int n = 0; n < 4; ++n) {  // n for 4 fragement

    // get row col for fragb
    int row_start = ik * 16;
    int col_start = tz * 64 + n * 16;

    for (int iter_load = 0; iter_load < 2; ++iter_load) {  // iter for 2 fragb load

      for (int i = 0; i < 4; i = i + 2) {  // i = 0 2
        int row{0}, col{0};
        int groupID = lane_id >> 2;
        int threadID_in_group = lane_id % 4;
        
        if (i < 2) {
          row = (threadID_in_group * 2) + (i & 0x1);
        } else {
          row = (threadID_in_group * 2) + (i & 0x1) + 8;
        }
        col = groupID;
        // if (tid == 64) {
        //   printf("tid %d, lane_id %d, row %d, col %d\n", tid, lane_id, row, col);
        // }

        row = row_start + row;
        col = col_start + col + iter_load * 8;

        // if (tid == 64) {
        //   printf("(%d, %d)= %f.\n ", row, col, __half2float(smem[row + col * 32]));
        // }

        frag[n * 4 + i/2 + iter_load * 2] = *(reinterpret_cast<unsigned int *>(smem + col * 32 + row));  // once load 2 half data
      }
    }
  }  // iter
}

template <int BM, int BN, int BK, int WMMA_M, int WMMA_N, int WMMA_K>
__device__ void store_fragc(float* smemc, float* frag_c, int frag_a_num, int frag_b_num){
  const int tid = threadIdx.x + threadIdx.y * 32 + threadIdx.z * 32 * 2;
  int lane_id = tid % 32;

  int srow = threadIdx.y * WMMA_M * frag_a_num;
  int scol = threadIdx.z * WMMA_N * frag_b_num;
  // float* smemec_ptr = smemc + + col * WMMA_N * frag_b_num;

  #pragma unroll
  for (int r = 0; r < frag_a_num; ++r) {
#pragma unroll
    for (int c = 0; c < frag_b_num; ++c) {

      for (int i = 0; i < 8; i++) {
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

        srow = srow + row + r * 16;
        scol = scol + col + c * 16;
        if (tid == 0) {
          printf("(%d, %d)=%f", srow, scol, frag_c[i]);
        }
        // smemc[srow *BN + scol] = frag_c[i];
      }
    }  // frag b
  }  // frag a
}

template <int BM, int BN, int BK>
__device__ void store2globalc(float* smemc, float* gmemc, int bx, int by, int N){
  const int row = threadIdx.x + threadIdx.y * 32 + threadIdx.z * 2 * 32;
  for (int col = 0; col < BN; ++col) {
      gmemc[(by * BM + row) * N + bx * BN + col] = smemc[row * BN + col];
      // gmemc[(by * BM + row) * N + bx * BN + col] = __float2half(smemc[row * BN + col]);
  }
}

template <int WMMA_M, int WMMA_N, int WMMA_K>
__device__ void print_fraga(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major>& frag, half* data) {
  if (threadIdx.y == 0 && threadIdx.z == 0 && threadIdx.x < 32) {
    const int tid = threadIdx.x + threadIdx.y * 32 + threadIdx.z * 32 * 2;
    int lane_id = tid % 32;
  
    // __shared__ half smem[smem_num];
    // half* s_a = smem;
    for (int i = 0; i < 8; ++i) {
      int row{0}, col{0};
      int groupID = lane_id >> 2;
      int threadID_in_group = lane_id % 4;
      
      if (i < 2 || (i >=4 && i < 6)) {
        row = groupID;
      } else {
        row = groupID + 8;
      }
      if (i < 4) {
        col = (threadID_in_group * 2) + (i & 0x1);
      } else {
        col = (threadID_in_group * 2) + (i & 0x1) + 8;
      }
      printf("tid = %d lane_id = %d, groupID = %d, i = %d, row col = (%d, %d) \n", tid, lane_id, groupID, i, row, col);
    }
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


__device__ void print_fragc(float* frag) {
  const int tid = threadIdx.x + threadIdx.y * 32 + threadIdx.z * 32 * 2;
  int lane_id = tid % 32;

  // __shared__ half smem[smem_num];
  // half* s_a = smem;
  for (int i = 0; i < 8; i++) {
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
    printf("tid = %d row col = (%d, %d), v = %f \n", tid, row, col, frag[i]);

  }
}


__device__ void mma_sync(unsigned int *frag_a, unsigned int *frag_b, float *frag_c)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=f"(frag_c[0]), "=f"(frag_c[1]), "=f"(frag_c[4]), "=f"(frag_c[5])
        : "r"(frag_a[0]), "r"(frag_a[2]), "r"(frag_a[1]), "r"(frag_a[3]),
          "r"(frag_b[0]), "r"(frag_b[1]),
          "f"(frag_c[0]), "f"(frag_c[1]), "f"(frag_c[4]), "f"(frag_c[5]));

    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=f"(frag_c[2]), "=f"(frag_c[3]), "=f"(frag_c[6]), "=f"(frag_c[7])
        : "r"(frag_a[0]), "r"(frag_a[2]), "r"(frag_a[1]), "r"(frag_a[3]),
          "r"(frag_b[2]), "r"(frag_b[3]),
          "f"(frag_c[2]), "f"(frag_c[3]), "f"(frag_c[6]), "f"(frag_c[7]));
}

template <int BM, int BN, int BK, int WARP_M, int WARP_N, int WARP_K, int WMMA_M, int WMMA_N, int WMMA_K, int THREADNUM, int STAGE>
__global__ void multi_stage2_mma_half_kernel(
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

  constexpr int smem_num = (BM * BK + BN * BK) * STAGE > 2 * BM * BN ? (BM * BK + BN * BK) * STAGE : 2 * BM * BN; // 2 for float, fix
  constexpr size_t smem_size = smem_num * sizeof(half);

  static_assert(smem_size < (48 << 10) && "smem_num must be small 49152 bytes.");

  __shared__ half smem[smem_num];
  half* smema = smem;
  half* smemb = smem + BM * BK;
  // half* smema2 = smemb + BN * BK;
  // half* smemb2 = smema2 + BM * BK;
  // half* smema3 = smemb2 + BN * BK;
  // half* smemb3 = smema3 + BM * BK;
  float* smemc = reinterpret_cast<float *>(smem); // 32768

  // prologue
  load_smema<BM, BN, BK>(smema, a + load_block_a_gmem_addr, K, 0);
  load_smemb<BM, BN, BK>(smemb, b + load_block_b_gmem_addr, K, 0);

  asm volatile("cp.async.commit_group;\n" ::);

  // load_smema<BM, BN, BK>(smema2, a + load_block_a_gmem_addr, K, 1);
  // load_smemb<BM, BN, BK>(smemb2, b + load_block_b_gmem_addr, K, 1);

  // asm volatile("cp.async.commit_group;\n" ::);

  // half *SA[2] = {smema, smema2};
  // half *SB[2] = {smemb, smemb2};

  constexpr int frag_a_num = WARP_M / WMMA_M;  // 64 / 16 = 4
  constexpr int frag_b_num = WARP_N / WMMA_N;  // 32 / 16 = 2

  unsigned int frag_a[4 * 4];  // first 4: 4 *2 = 8 half for one thread, second 4: 1 warp compute 4 warp, so 1 thread need 4 register load
  unsigned int frag_b[4 * 4];

  float frag_c[128] = {0.0};  // one thread store 4 * 4, total 8 frage c 

  // if (threadIdx.y == 0 && threadIdx.z == 0 && threadIdx.x < 32) {
  //   const int tid = threadIdx.x + threadIdx.y * 32 + threadIdx.z * 32 * 2;
  //   int lane_id = tid % 32;
  
  //   // __shared__ half smem[smem_num];
  //   // half* s_a = smem;
  //   for (int i = 0; i < 8; ++i) {
  //     int row{0}, col{0};
  //     int groupID = lane_id >> 2;
  //     int threadID_in_group = lane_id % 4;
      
  //     if (i < 2 || (i >=4 && i < 6)) {
  //       row = groupID;
  //     } else {
  //       row = groupID + 8;
  //     }
  //     if (i < 4) {
  //       col = (threadID_in_group * 2) + (i & 0x1);
  //     } else {
  //       col = (threadID_in_group * 2) + (i & 0x1) + 8;
  //     }
  //     printf("tid = %d lane_id = %d, groupID = %d, i = %d, row col = (%d, %d) \n", tid, lane_id, groupID, i, row, col);
  //   }
  
  // }
  int loop_num = K / BK;

  // lk = 0 -> 0, 1; lk = 2 -> 2, 3; ... lk = 62 -> 62, 63
#pragma unroll
  for (int lk = 0; lk < loop_num; ++lk) { // do loop for lk, lk = 0-> SA[0], lk = 1 -> SA[1]
    asm volatile("cp.async.wait_group %0;\n" ::"n"(1));  // wait smema [smema, smema2]
    __syncthreads();

    #pragma unroll
    for (int ik = 0; ik < BK / WARP_K; ik += 1) {
      load_fraga<BM, BN, BK, WMMA_M, WMMA_N, WMMA_K>(smema, frag_a, ik);
      load_fragb<BM, BN, BK, WMMA_M, WMMA_N, WMMA_K>(smemb, frag_b, ik);
      mma_sync(frag_a, frag_b, frag_c);
      // print_fragc(frag_c);
  
    }

  }
  __syncthreads();
  store_fragc<BM, BN, BK, WMMA_M, WMMA_N, WMMA_K>(smemc, frag_c, frag_a_num, frag_b_num);
  // __syncthreads();
  // store2globalc<BM, BN, BK>(smemc, c, bx, by, N);
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

    // cudaDeviceProp prop;
    // cudaGetDeviceProperties(&prop, 0);

    // // 每个Block最大可用共享内存（静态+动态）
    // printf("每个Block最大共享内存: %zu byte\n", 
    //        prop.sharedMemPerBlock);

    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 32;
    // A &&B (128 * 32 + 64 * 32） * 2（half) * 3(stage) = (4096 + 2048) * 6 = 36864
    // shared C: 128 * 64 * 4 = 32768

    constexpr int WARP_M = 64;
    constexpr int WARP_N = 64;
    constexpr int WARP_K = 16;

    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;
    constexpr int thread_num = 32 * 4; // 4 warp
    constexpr int stage = 1; // 3 pingpong pipeline

    dim3 GridDim(DIV_UP(N, BN), DIV_UP(M, BM));
    dim3 BlockDim(32, 1, 1);

    multi_stage2_mma_half_kernel<BM, BN, BK, WARP_M, WARP_N, WARP_K, WMMA_M, WMMA_N, WMMA_K, thread_num, stage>
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
