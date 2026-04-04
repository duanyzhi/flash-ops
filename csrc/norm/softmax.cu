#include "cuda_utils.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/AccumulateType.h>
#include <iostream>
#include <cuda_fp16.h>
#include "../ops.h"

namespace flash_ops {

__device__ void warp_max(float *data) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float pval = data[tid];

  pval = max(pval, __shfl_xor_sync(0xffffffff, pval, 16, 32));
  pval = max(pval, __shfl_xor_sync(0xffffffff, pval, 8, 32));
  pval = max(pval, __shfl_xor_sync(0xffffffff, pval, 4, 32));
  pval = max(pval, __shfl_xor_sync(0xffffffff, pval, 2, 32));
  pval = max(pval, __shfl_xor_sync(0xffffffff, pval, 1, 32));

  __syncthreads();
  data[tid] = pval;
}

__device__ void warp_sum(float *data) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float pval = data[tid];

  pval += __shfl_xor_sync(0xffffffff, pval, 16, 32);
  pval += __shfl_xor_sync(0xffffffff, pval, 8, 32);
  pval += __shfl_xor_sync(0xffffffff, pval, 4, 32);
  pval += __shfl_xor_sync(0xffffffff, pval, 2, 32);
  pval += __shfl_xor_sync(0xffffffff, pval, 1, 32);
  __syncthreads();
  data[tid] = pval;
}

template<int thread_num>
__global__ void forward_kernel(const float* __restrict__ x, float* output, int B, int L) {
  int tid = blockIdx.x + blockDim.x + threadIdx.x;
  const float* block_x = x + blockIdx.x * L;
  // float* block_y = output + blockIdx.x * L;

  // loop-1. block max x
  __shared__ float max_vector[thread_num];

  int num = L / thread_num;
  float max_value = block_x[threadIdx.x];
  for (int i = 1; i < num; ++i) {
    max_value = max(max_value, block_x[threadIdx.x * num + i]);
  }
  max_vector[threadIdx.x] = max_value;

  warp_max(max_vector);

  float mV = 0;
  if (threadIdx.x == 0) {
    printf("mV=%f\n", max_vector[0]);
    mV = max_vector[0];
  }

  // loop-2. sum all 
  // d_j = d_j-1 + exp(xj-mV)
  __shared__ float sum_dj[thread_num];
  float sum_d_j = 0;
  for (int j = 0; j < num; ++j) {
    sum_d_j += __expf(block_x[threadIdx.x * num + j] - mV);
  }
  sum_dj[threadIdx.x] = sum_d_j; 
  
  __syncthreads();
  warp_sum(sum_dj);

  float sum_scale = 1.0 / sum_dj[0];;
  // if (threadIdx.x == 0) {
  //   sum_scale = 1.0 / sum_dj[0];
  //   printf("sum=%f\n", sum_scale);
  // }

  // loop-3: y_j = exp(x_j - mV) / sum_d_j
  for (int j = 0; j < num; ++j) {
    // printf("id = %d, value=%f, exp=%f, output_id=%d, sum=%f\n", threadIdx.x * num + j, block_x[threadIdx.x * num + j], __expf(block_x[threadIdx.x * num + j] - mV), threadIdx.x * num + j + blockIdx.x * L, __expf(block_x[threadIdx.x * num + j] - mV) * sum_scale);;
    // printf("exp=%f, exp * scale=%f\n", __expf(block_x[threadIdx.x * num + j] - mV), __expf(block_x[threadIdx.x * num + j] - mV) * sum_scale);

    output[threadIdx.x * num + j + blockIdx.x * L] = __expf(block_x[threadIdx.x * num + j] - mV) * sum_scale;
  }
  __syncthreads();
}

at::Tensor softmax(const at::Tensor& input) {
    const int B = input.size(0);
    const int L = input.size(1);
    at::Tensor output = at::zeros(
        {B, L}, 
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)
    );

    constexpr int thread_num = 32;
    dim3 grid_dim(B);
    dim3 block_dim(thread_num);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    forward_kernel<thread_num><<<grid_dim, block_dim, 0, stream>>>(
      reinterpret_cast<float*>(input.data_ptr()),
      reinterpret_cast<float*>(output.data_ptr()),
      B, L
    );

    cudaStreamSynchronize(stream);

    return output;
}

} // namespace flash_ops