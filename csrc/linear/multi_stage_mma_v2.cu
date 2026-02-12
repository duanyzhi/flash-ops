#include <cuda_fp16.h>
#include <mma.h>
#include "gemm.h"
#include "cuda_utils.cuh"

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

namespace flash_ops {

#define MAX(a, b) (a) > (b) ? (a) : (b)

template <int BM, int BN, int BK, int BK_PAD>
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
    
    void* ptr = (void*)(smema + row * BK_PAD + col);
    uint32_t smem_ptr; 
    asm(
      "{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }\n"
      : "=r"(smem_ptr) : "l"(ptr)
    );
    asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" :: "r"(smem_ptr),
                 "l"(&a[row * K + col + ik * BK]), "n"(16));
  }
}

template <int BM, int BN, int BK, int BK_PAD>
__device__ void load_smemb(half* smemb, const half* __restrict__ b, int K, int ik) {
  // load b: [64, 32] [N, K], colmajor
  int tid = threadIdx.x + threadIdx.y * 32 + threadIdx.z * 2 * 32;
  
  for (int i = 0; i < 4; ++i) {
    int col = i * 32 + tid / 4;
    int row = tid % 4 * 8;

    // col = col ^ (((row & 3) << 3));
  
    // if (ik == 1) {
    //   printf("row = %d, col = %d, ik = %d, smem=%d,  gmem=%d ;;;", row, col, ik, col * BK_PAD + row, col * K + row + ik * BK);
    // }
    // if (col * K + row + ik * BK > 128  * 64) {
    //   printf("error ..... row=%d, col=%d, K=%d, ik = %d, BK=%d\n", row, col, K, ik, BK);
    // }

    void* ptr = (void*)(smemb + col * BK_PAD + row);
    uint32_t smem_ptr; 
    asm(
      "{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }\n"
      : "=r"(smem_ptr) : "l"(ptr)
    );

    asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" :: "r"(smem_ptr),
                 "l"(&b[col * K + row + ik * BK]), "n"(16));
    
  }
}

template <int BM, int BN, int BK, int BK_PAD, int WMMA_M, int WMMA_N, int WMMA_K>
__device__ void load_fraga(half *smem, unsigned int *frag, int ik){
  // load 64x16 frag

  // T0: (0,0),(0,1),(8,0),(8,1),(0,8),(0,9),(8,8),(8,9)
  // T1: (1,0),(1,1),(9,0),(9,1),(1,8),(1,9),(9,8),(9,9)
  // T2: (2,0),(2,1),(10,0),(10,1),(2,8),(2,9),(10,8),(10,9)
  //　...

  const int tid_y = threadIdx.y;
  const int tid_z = threadIdx.z;
  
  const int tid = threadIdx.x + tid_y * 32 + tid_z * 32 * 2;
  constexpr int warp_size = 32;
  int wid_id = tid / warp_size;
  int lane_id = tid % warp_size;

  int smem_row = threadIdx.y * 64;

  for (int n = 0; n < 4; ++n) {  // n for 4 fragement
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

      row = smem_row + n * 16 + row;
      col = ik * 16 + col;
      
      frag[n * 4 + i/2] = *(reinterpret_cast<unsigned int *>(smem + row * BK_PAD + col));

    }
  }
}

template <int BM, int BN, int BK, int BK_PAD, int WMMA_M, int WMMA_N, int WMMA_K>
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

        row = row_start + row;
        col = col_start + col + iter_load * 8;

        frag[n * 4 + i/2 + iter_load * 2] = *(reinterpret_cast<unsigned int *>(smem + col * BK_PAD + row));  // once load 2 half data
      }
    }
  }  // iter
}

template <int BM, int BN, int BK, int WMMA_M, int WMMA_N, int WMMA_K>
__device__ void store_fragc(half* smemc, float* frag_c, int frag_a_num, int frag_b_num){
  // write 128 ele for one thread

  const int tid = threadIdx.x + threadIdx.y * 32 + threadIdx.z * 32 * 2;
  int lane_id = tid % 32;

  int start = threadIdx.y * WMMA_M * frag_a_num;
  int end = threadIdx.z * WMMA_N * frag_b_num;

  int groupID = lane_id >> 2;
  int tid4 = lane_id & 3;
  int base_col = tid4 * 2;

  int row0 = groupID;
  int col0 = base_col;

  int row2 = groupID + 8;
  int col2 = base_col;

  int row4 = groupID;
  int col4 = base_col + 8;

  int row6 = groupID + 8;
  int col6 = base_col + 8;

  int BN_PAD = BN + 8;  // pad for store bank conflict
  // constexpr int BN_half = BN / 2;

  // float2* frag_c_float2 = reinterpret_cast<float2*>(frag_c);

#pragma unroll
  for (int r = 0; r < frag_a_num; ++r) {
#pragma unroll
    for (int c = 0; c < frag_b_num; ++c) {
      // (0, 0), (0, 1)
      int srow0 = start + row0 + r * 16;
      int scol0 = end + col0 + c * 16;
      reinterpret_cast<half2*>(smemc)[(srow0 * BN_PAD + scol0)/2] = __float22half2_rn(reinterpret_cast<float2*>(frag_c)[(0 + c * 8 + r * 32) / 2]);

      // smemc[srow0 * BN + scol0] = frag_c[0 + c * 8 + r * 32];
      // smemc[srow0 * BN + scol0 + 1] = frag_c[1 + c * 8 + r * 32];

      // (8, 0), (8, 1)
      int srow2 = start + row2 + r * 16;
      int scol2 = end + col2 + c * 16;
      reinterpret_cast<half2*>(smemc)[(srow2 * BN_PAD + scol2)/2] = __float22half2_rn(reinterpret_cast<float2*>(frag_c)[(2 + c * 8 + r * 32) / 2]);
      // smemc[srow2 * BN + scol2 + 1] = frag_c[3 + c * 8 + r * 32];

      // (0, 8), (0, 9)
      int srow4 = start + row4 + r * 16;
      int scol4 = end + col4 + c * 16;
      reinterpret_cast<half2*>(smemc)[(srow4 * BN_PAD + scol4)/2] = __float22half2_rn(reinterpret_cast<float2*>(frag_c)[(4 + c * 8 + r * 32) / 2]);

      // smemc[srow4 * BN + scol4] = frag_c[4 + c * 8 + r * 32];
      // smemc[srow4 * BN + scol4 + 1] = frag_c[5 + c * 8 + r * 32];

      // (8, 8), (8, 9)
      int srow6 = start + row6 + r * 16;
      int scol6 = end + col6 + c * 16;
      reinterpret_cast<half2*>(smemc)[(srow6 * BN_PAD + scol6)/2] = __float22half2_rn(reinterpret_cast<float2*>(frag_c)[(6 + c * 8 + r * 32) / 2]);

      // smemc[srow6 * BN + scol6] = frag_c[6 + c * 8 + r * 32];
      // smemc[srow6 * BN + scol6 + 1] = frag_c[7 + c * 8 + r * 32];

      // if (tid < 32 && r == 0 && c == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
      //   printf("lane id = %d, (%d,%d), (%d,%d), (%d, %d), (%d, %d)\n", lane_id, row0, col0, row2, col2, row4, col4, row6, col6);
      // }

      // for (int i = 0; i < 8; i++) {
      //   int row{0}, col{0};
      //   int groupID = lane_id >> 2;
      //   int threadID_in_group = lane_id % 4;
      //   if (i < 4) {
      //     row = i < 2 ? groupID : groupID + 8;
      //     col = (threadID_in_group * 2) + (i & 0x1);
      //   } else {
      //     int j = i - 4;
      //     row = j < 2 ? groupID : groupID + 8;
      //     col = (threadID_in_group * 2) + (j & 0x1) + 8;
      //   }

      //   int srow = start + row + r * 16;
      //   int scol = end + col + c * 16;

      //   // scol ^= ((srow & 7) << 3);

      //   smemc[srow * BN + scol] = frag_c[i + c * 8 + r * 32];
      // }
    }  // frag b
  }  // frag a
}

template <int BM, int BN, int BK>
__device__ void store2globalc(half* smemc, half* gmemc, int bx, int by, int N){
  const int tid = threadIdx.x + threadIdx.y * 32 + threadIdx.z * 32 * 2;
  constexpr int warp_size = 32;
  int wid_id = tid / warp_size;
  int lane_id = tid % warp_size;
  
  int row_start = wid_id * 32;

  int col = lane_id;
  int col2 = lane_id + 32;

  int N_half = N / 2;

  int BN_PAD = BN + 8;
  constexpr int BN_half = BN / 2;

  #pragma unroll
  for (int r = 0; r < 32; ++r) {
    int row = row_start + r;
    reinterpret_cast<half2*>(gmemc)[(by * BM + row) * N_half + bx * BN_half + col] = reinterpret_cast<half2*>(smemc)[row *  BN_PAD / 2 + col];
    reinterpret_cast<half2*>(gmemc)[(by * BM + row) * N_half + bx * BN_half + col2] = reinterpret_cast<half2*>(smemc)[row * BN_PAD / 2 + col2];
  }
}

// do one mma
__device__ void mma_sync(unsigned int *frag_a, unsigned int *frag_b, float *frag_c)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=f"(frag_c[0]), "=f"(frag_c[1]), "=f"(frag_c[2]), "=f"(frag_c[3])
        : "r"(frag_a[0]), "r"(frag_a[1]), "r"(frag_a[2]), "r"(frag_a[3]),
          "r"(frag_b[0]), "r"(frag_b[1]),
          "f"(frag_c[0]), "f"(frag_c[1]), "f"(frag_c[2]), "f"(frag_c[3]));

    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=f"(frag_c[4]), "=f"(frag_c[5]), "=f"(frag_c[6]), "=f"(frag_c[7])
        : "r"(frag_a[0]), "r"(frag_a[1]), "r"(frag_a[2]), "r"(frag_a[3]),
          "r"(frag_b[2]), "r"(frag_b[3]),
          "f"(frag_c[4]), "f"(frag_c[5]), "f"(frag_c[6]), "f"(frag_c[7]));
}

template <int BM, int BN, int BK, int WARP_M, int WARP_N, int WARP_K, int WMMA_M, int WMMA_N, int WMMA_K, int THREADNUM, int STAGE>
__global__ void multi_stage2_mma_half_kernel(
    const half* __restrict__ a, const half* __restrict__ b,
    const half* __restrict__ bias, half* __restrict__ c, const int M,
    const int N, const int K) {
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  const int tid_y = threadIdx.y;
  const int tid_z = threadIdx.z;

  const int tid = threadIdx.x + tid_y * 32 + tid_z * 32 * 2;
  constexpr int warp_size = 32;
  int wid_id = tid / warp_size;
  int lane_id = tid % warp_size;

  int load_block_a_gmem_addr = by * BM * K;
  int load_block_b_gmem_addr = bx * BN * K;

  // printf("bx, by=%d,%d",bx,by);

  constexpr int BK_PAD = BK + 8;

  constexpr int smem_num = (BM * BK_PAD + BN * BK_PAD) * STAGE > BM * BN ? (BM * BK_PAD + BN * BK_PAD) * STAGE : BM * BN;
  constexpr size_t smem_size = smem_num * sizeof(half);

  static_assert(smem_size < (48 << 10) && "smem_num must be small 49152 bytes.");

  __shared__ half smem[smem_num];
  half* smema = smem;
  half* smemb = smem + BM * BK_PAD;
  half* smema2 = smemb + BN * BK_PAD;
  half* smemb2 = smema2 + BM * BK_PAD;

  half* smemc = smem;

  // // prologue
  load_smema<BM, BN, BK, BK_PAD>(smema, a + load_block_a_gmem_addr, K, 0);
  load_smemb<BM, BN, BK, BK_PAD>(smemb, b + load_block_b_gmem_addr, K, 0);

  asm volatile("cp.async.commit_group;\n" ::);

  // load_smema<BM, BN, BK>(smema2, a + load_block_a_gmem_addr, K, 1);
  // load_smemb<BM, BN, BK>(smemb2, b + load_block_b_gmem_addr, K, 1);

  // asm volatile("cp.async.commit_group;\n" ::);

  half *SA[2] = {smema, smema2};
  half *SB[2] = {smemb, smemb2};

  constexpr int frag_a_num = WARP_M / WMMA_M;  // 64 / 16 = 4
  constexpr int frag_b_num = WARP_N / WMMA_N;  // 32 / 16 = 2

  unsigned int frag_a[4 * 4];  // first 4: 4 *2 = 8 half for one thread, second 4: 1 warp compute 4 frag, so 1 thread need 4 register load
  unsigned int frag_b[4 * 4];

  // 16 * 16 / 32 = 8
  // notice: one warp
  float frag_c[128] = {0.0};  // one thread store 4 * 4, total 8 frage c， 

#pragma unroll
  for (int lk = 0; lk < K / BK; ++lk) { // do loop for lk, lk = 0-> SA[0], lk = 1 -> SA[1]
    // load next
    if (lk + 1 < K / BK) {
      load_smema<BM, BN, BK, BK_PAD>(SA[(lk + 1) % 2], a + load_block_a_gmem_addr, K, lk + 1);
      load_smemb<BM, BN, BK, BK_PAD>(SB[(lk + 1) % 2], b + load_block_b_gmem_addr, K, lk + 1);
    }
    asm volatile("cp.async.commit_group;\n" ::);

    // compute
    asm volatile("cp.async.wait_group %0;\n" ::"n"(1));
    __syncthreads();

    // // // prologue for stage 1
    // load_smema<BM, BN, BK, BK_PAD>(smema, a + load_block_a_gmem_addr, K, lk);
    // load_smemb<BM, BN, BK, BK_PAD>(smemb, b + load_block_b_gmem_addr, K, lk);

    // asm volatile("cp.async.commit_group;\n" ::);
    // asm volatile("cp.async.wait_group %0;\n" ::"n"(0));
    // __syncthreads();

    #pragma unroll
    for (int ik = 0; ik < WARP_K / WMMA_K; ik += 1) {
      load_fraga<BM, BN, BK, BK_PAD, WMMA_M, WMMA_N, WMMA_K>(SA[lk % 2], frag_a, ik);
      load_fragb<BM, BN, BK, BK_PAD, WMMA_M, WMMA_N, WMMA_K>(SB[lk % 2], frag_b, ik);
      // load_fraga<BM, BN, BK, BK_PAD, WMMA_M, WMMA_N, WMMA_K>(smema, frag_a, ik);
      // load_fragb<BM, BN, BK, BK_PAD, WMMA_M, WMMA_N, WMMA_K>(smemb, frag_b, ik);

      for (int mii = 0; mii < WARP_M / WMMA_M; mii += 1) { // 0，1，2，3 frag_a
        for (int nii = 0; nii < WARP_N / WMMA_N; nii += 1) {  // 0，1，2，3 frag_b
          mma_sync(&frag_a[mii*4], &frag_b[nii*4], &frag_c[nii*8 + mii*4*8]);
        }
      } 
    }
  }
 
  __syncthreads();

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
        // torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)
        torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA)
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
    constexpr int WARP_K = BK;

    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;
    constexpr int thread_num = 32; // 4 warp
    constexpr int stage = 2; // 3 pingpong pipeline

    dim3 GridDim(DIV_UP(N, BN), DIV_UP(M, BM));
    dim3 BlockDim(32, 2, 2);

    multi_stage2_mma_half_kernel<BM, BN, BK, WARP_M, WARP_N, WARP_K, WMMA_M, WMMA_N, WMMA_K, thread_num, stage>
        <<<GridDim, BlockDim, 0, stream>>>(
               reinterpret_cast<__half*>(input.data_ptr<at::Half>()),
               reinterpret_cast<__half*>(weight.data_ptr<at::Half>()),
               nullptr,
               reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
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
