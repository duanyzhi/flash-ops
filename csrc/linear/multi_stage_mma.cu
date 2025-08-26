#include <cuda_fp16.h>
#include <mma.h>
#include "gemm.h"
#include "cuda_utils.cuh"

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

namespace flash_ops {

template <int BM, int BN, int BK>
__device__ void load_smema(half* smema, const half* __restrict__ a, int K) {
  // load a: [128, 32] [M, K], rowmajor
  // tid = 0 load 8 ele, row = 0, col = 0...7
  // tid = 1 load 8 ele, row = 0, col = 8...15
  // tid = 2 load 8 ele, row = 0, col = 16...23
  // tid = 3 load 8 ele, row = 0, col = 24...31
  // tid = 4 load 8 ele, row = 1, col = 0...7
  // ....
  int tid = threadIdx.x; // thread num = 128

  for (int i = 0; i < 4; ++i) {
    int row = i * 32 + tid / 4;
    int col = tid % 4 * 8;
    // printf("row %d col: %d", row, col);
  
    void* ptr = (void*)(smema + row * BK + col);
    uint32_t smem_ptr; 
    asm(
      "{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }\n"
      : "=r"(smem_ptr) : "l"(ptr)
    );
    asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" :: "r"(smem_ptr),
                 "l"(&a[row * K + col]), "n"(16));
  }
}

template <int BM, int BN, int BK>
__device__ void load_smemb(half* smemb, const half* __restrict__ b, int K) {
  // load b: [128, 32] [N, K], colmajor
  int tid = threadIdx.x; // thread num = 128
  
  for (int i = 0; i < 4; ++i) {
    int row = i * 32 + tid / 4;
    int col = tid % 4 * 8;
  
    void* ptr = (void*)(smemb + row * BK + col);
    uint32_t smem_ptr; 
    asm(
      "{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }\n"
      : "=r"(smem_ptr) : "l"(ptr)
    );
    asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" :: "r"(smem_ptr),
                 "l"(&b[row * K + col]), "n"(16));
  }
}

template <int BM, int BN, int BK>
__device__ void load_fraga(half* smema, fragement_a_type frag_a){

}

template <int BM, int BN, int BK, int WARP_M, int WARP_N, int WARP_K, int WMMA_M, int WMMA_N, int WMMA_K, int THREADNUM>
__global__ void multi_stage_mma_half_kernel(
    const half* __restrict__ a, const half* __restrict__ b,
    const half* __restrict__ bias, float* __restrict__ c, const int M,
    const int N, const int K) {
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  const int tid = threadIdx.x;
  constexpr int warp_size = 32;
  int wid = tid / warp_size;
  int lane_id = tid % warp_size;

  constexpr int warp_num = 4;


  int load_block_a_gmem_addr = by * BM * K;
  int load_block_b_gmem_addr = bx * BN * K;

  constexpr int smem_size = BM * BK + BN * BK;
  __shared__ half smem[smem_size];
  half* smema = smem;
  half* smemb = smem + BM * BK;

  load_smema<BM, BN, BK>(smema, a + load_block_a_gmem_addr, K);
  load_smema<BM, BN, BK>(smemb, b + load_block_b_gmem_addr, K);

  asm volatile("cp.async.commit_group;\n" ::);
  asm volatile("cp.async.wait_group 0;\n" ::);

  __syncthreads(); 
  // if (tid == 0) {
  //   for (int i = 0; i < 32; ++i) {
  //     printf("%f ", __half2float(smemb[i]));
  //   }
  //   printf("\n");
  // }

  constexpr int frag_a_num = WARP_M / WMMA_M;
  constexpr int frag_b_num = WARP_N / WMMA_N;
  // nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> frag_a[frag_a_num];
  fragement_a_type frag_a[frag_a_num];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::col_major> frag_b[frag_b_num];
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_c[frag_a_num][frag_b_num];

  for (int ik = 0; ik < K / BK; ++ik) {
    // for loop for mma
    // one wrap 64x64x16, one mma 16x16x16
    for (int mk_row = 0; mk_row < 4; ++mk_row) {
      for (int mk_col = 0; mk_col < 4; ++mk_col) {
      }
    }
  }

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
    );
    // at::Tensor c = at::zeros(output_size, input.options());

    // block tile (BM, BN, BK) = (128, 128, 32)
    // warp tile (WARP_M, WARP_N, WARP_K) = (64, 64, 16)
    // mma tile (WMMA_M, WMMA_N, WMMA_K) = (16, 16, 16)
    // one warp process 64 * 64 * 16, so need 4 warp
    // one warp need 64 / 16 * 64 / 16 * 16 / 16 = 16 time wmma

    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 32;

    constexpr int WARP_M = 64;
    constexpr int WARP_N = 64;
    constexpr int WARP_K = 16;

    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;
    constexpr int thread_num = 32 * 4; // 4 warp

    dim3 GridDim(DIV_UP(N, BN), DIV_UP(M, BM));  // 8 * 8
    dim3 BlockDim(thread_num, 1);
    float* c_ptr = reinterpret_cast<float*>(output.data_ptr());

    using fragement_a_type = nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major>;
    multi_stage_mma_half_kernel<BM, BN, BK, WARP_M, WARP_N, WARP_K, WMMA_M, WMMA_N, WMMA_K, thread_num>
        <<<GridDim, BlockDim, 0, stream>>>(
               reinterpret_cast<__half*>(input.data_ptr<at::Half>()),
               reinterpret_cast<__half*>(weight.data_ptr<at::Half>()),
               nullptr,
               c_ptr,
               M, N, K);

    cudaStreamSynchronize(stream);
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
    return output;
}


} // flash_ops
