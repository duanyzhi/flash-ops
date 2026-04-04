#include <cuda_fp16.h>
#include <mma.h>
#include "../ops.h"

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

namespace flash_ops {

template<int head_dim, int thread_num>
__gobal__ void sim_mm_kernel_impl(const half* __restrict__ q, const half* __restrict__ k, const half* __restrict__ v,
   half* __restrict__ o, int B, int num_head, int N, int head_dim, int len_thread) {

    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    int tid = threadIdx.x;

    printf("bx, by: %d, %d\n ", bx, by);
    int tile_size = len_thread * head_dim;

    __shared__ half smem[tile_size + tile_size];

    half* q_smem = smem;
    half* k_smem = smem + tile_size;

    int start_global_qk = tid * len_thread * head_dim;

    // 1. load global to memroy
    for (int i = 0; i < len_thread; ++i) {
        for (int d = 0; d < head_dim; ++d) {
            q_smem[i * head_dim + d] = q[start_global_qk + i * head_dim + d];
            k_smem[i * head_dim + d] = k[start_global_qk + i * head_dim + d];
        }
    }

    // 2. compute
    for (int x = 0; x < len_thread; ++x) {
        for (int y = 0; y < head_dim; ++y) {
            q_smem[x * head_dim + y] * k_smem[x * head_dim + y]
        }
    }
}

at::Tensor simple_gemm(const at::Tensor & A, const at::Tensor & B, onst c10::optional<at::Tensor> & bias) {
    const M = A.size(0);
    const K = A.size(1);
    const N = B.size(1);

    constexpr int thread_num = 32;

    const len_thread = N / thread_num;  // 1 thread process continue len_thread row

    at::Tensor output = at::zeros(
        {M, N}, 
        torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA)
    );

    dim3 grid_dim(B, num_head);  // batch_size x num_heads
    dim3 block_dim(thread_num);  // threads per block

    sim_mm_kernel_impl<thread_num>
        <<<GridDim, BlockDim, 0, stream>>>(
               reinterpret_cast<__half*>(Q.data_ptr<at::Half>()),
               reinterpret_cast<__half*>(K.data_ptr<at::Half>()),
               reinterpret_cast<__half*>(V.data_ptr<at::Half>()),
               reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
               B, num_head, N, head_dim, len_thread);

    cudaStreamSynchronize(stream);
    cudaError_t err = cudaGetLastError();
}

} // flash_ops