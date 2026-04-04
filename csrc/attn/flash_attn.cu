#include <cuda_fp16.h>
#include <mma.h>
#include "../ops.h"

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

namespace flash_ops {

template<int BM, int BN, int BK, int thread_num>
__global__ void naive_attn_forward_impl(const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
    float* __restrict__ o,  int B, int num_head, int seq_num, int head_dim, float softmax_scale) {
    const int bx = blockIdx.x;  // for batch size
    const int by = blockIdx.y;  // for num head

    // seq_num = BM = BN = 32 = N; head_dim = BK = 128

    // printf("bx = %d, by = %d\n", bx, by);
        
    // int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int batch_step = num_head * seq_num * head_dim;

    const float* block_q = q + bx * batch_step + by * seq_num * head_dim;
    const float* block_k = k + bx * batch_step + by * seq_num * head_dim;
    const float* block_v = v + bx * batch_step + by * seq_num * head_dim;

    // each thread process BK num, thread_num = BM = 32
    // for i 1; N do 
    //   xj = Qj * Kk
    //   mj = max (mj−1, xj)
    //   dj = dj−1 × e^(mj−1−mj) + e^(xj−mj)

    // // 1. load Q and K to sram
    constexpr int tile_size = BM * BK + BN * BK + BN * BK; // Q + K + V
    __shared__ float smem[tile_size];

    constexpr int sum_tile_size = BM * BN;
    __shared__ float attn_smem[sum_tile_size];

    float* q_smem = smem;
    float* k_smem = q_smem + BM * BK;
    float* v_smem = k_smem + BN * BK;

    int start_global_idx = threadIdx.x * BK;

    // 1. load global to memroy
#pragma unroll
    for (int d = 0; d < BK; ++d) {
        q_smem[start_global_idx + d] = block_q[start_global_idx + d];
        k_smem[start_global_idx + d] = block_k[start_global_idx + d];
        v_smem[start_global_idx + d] = block_v[start_global_idx + d];
    }

    __syncthreads(); 

    // Q * K, each thread compute BN num output
    // oneline softmax by row(j)
    for (int j = 0; j < BN; ++j) {
        float sum = 0;
        for (int ik = 0; ik < BK; ++ik) {
          sum += q_smem[threadIdx.x * BK + ik] * k_smem[j * BK + ik];
        }
        // row is threadIdx.x, col = j
        attn_smem[threadIdx.x * BN + j] = sum * softmax_scale;
    }

    __syncthreads();

    // do online softmax
    float max_pre = attn_smem[threadIdx.x * BN];
    float mj = attn_smem[threadIdx.x * BN];
    float dj_1 = __expf(attn_smem[threadIdx.x * BN] - mj);
    float dj = 0;
    for (int j = 1; j < BN; ++j) {
      mj = max(max_pre, attn_smem[threadIdx.x * BN + j]);
      dj = dj_1 * __expf(max_pre - mj) + __expf(attn_smem[threadIdx.x * BN + j] - mj);
    
      // update
      max_pre = mj;
      dj_1 = dj;
    }

    __syncthreads();
    float sum_scale = 1.0 / dj;

    for (int j = 0; j < BN; ++j) {
        attn_smem[threadIdx.x * BN + j] = __expf(attn_smem[threadIdx.x * BN + j] - mj) * sum_scale;
    }
    __syncthreads();

    // A @ V
    for (int ik = 0; ik < BK; ++ik) { // ik is output col, threadIdx.x is output row
        float sum_av = 0.0;
        for (int j = 0; j < BN; ++j) { // j is for sum loop
            sum_av += attn_smem[threadIdx.x * BN + j] * v_smem[j * BK + ik];
        }
        o[bx * batch_step + by * seq_num * head_dim + threadIdx.x * BK + ik] = sum_av;
    }
    
}

at::Tensor attention(const at::Tensor & Q, const at::Tensor & K, const at::Tensor & V) {
    // std::cout << Q.sizes().vec() << "; " << K.sizes().vec() << "; " << V.sizes().vec() << "\n";
    // 1 8 32 128; 1 8 128 32; 1 8 32 128
    // batch_size = 1, n_head = 8, kv_head = 8, seq_len = 32, head_dim = 128

    const int B = Q.size(0);
    const int num_head = Q.size(1);

    const int seq_num = Q.size(2);   // seq = N = 32
    const int head_dim = Q.size(3);  // head_dim = 128
    
    constexpr int thread_num = 32;

    constexpr int BM = 32;
    constexpr int BN = 32;
    constexpr int BK = 64;

    const float softmax_scale = 1.0 / sqrt(head_dim);

    at::Tensor output = at::zeros(
        {B, num_head, seq_num, head_dim}, 
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)
    );

    dim3 grid_dim(B, num_head);  // batch_size x num_heads
    dim3 block_dim(thread_num);  // threads per block
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    naive_attn_forward_impl<BM, BN, BK, thread_num>
        <<<grid_dim, block_dim, 0, stream>>>(
               reinterpret_cast<float*>(Q.data_ptr()),
               reinterpret_cast<float*>(K.data_ptr()),
               reinterpret_cast<float*>(V.data_ptr()),
               reinterpret_cast<float*>(output.data_ptr()),
               B, num_head, seq_num, head_dim, softmax_scale);

    cudaStreamSynchronize(stream);
    cudaError_t err = cudaGetLastError();

    return output;
}

} // flash_ops