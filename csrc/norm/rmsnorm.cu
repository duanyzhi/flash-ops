#include "norm.h"
#include "cuda_utils.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/AccumulateType.h>
#include <iostream>
#include <cuda_fp16.h>

namespace flash_ops {

__global__ void rmsnorm_kernel(const half* __restrict__ x, half* __restrict__ output, int N) {
  int tid = threadIdx.x;
  const half* data = x + blockIdx.x * N;
  // printf("blockDimx: ", blockDim.x);
  // one thread process N / blockDimx.x ele
  int stride = blockDim.x;
  // printf("thread %d \n", tid);

  float sum_xx = 0;

#pragma unroll
  for (uint id = tid; id < N; id+=stride) {
    sum_xx += __half2float(__hmul(data[id], data[id]));
  }

  // warp sum
  float reduce_val = sum_xx;
  reduce_val = warp_reduce_sum(reduce_val);

  __shared__ float shared_sum[32];
  int lane_id = threadIdx.x % 32;
  int warp_id = threadIdx.x / 32;

  // write all warp data
  if (lane_id == 0) {
    shared_sum[warp_id] = reduce_val;
  }
  __syncthreads();

  if (threadIdx.x < 32) { // only use first warp
    reduce_val = shared_sum[lane_id];
  } else {
    reduce_val = 0.0f;
  }

  reduce_val = warp_reduce_sum(reduce_val);
  // compute sqrt
  __shared__ float sqrt_val;
  if (tid == 0) {
    sqrt_val = rsqrtf(__ddiv_rn(reduce_val, N));
  }

  __syncthreads();

  half* output_data = output + blockIdx.x * N;
#pragma unroll
  for (uint idx = tid; idx < N; idx+=blockDim.x) {
    output_data[idx] = data[idx] * __float2half(sqrt_val);
  }
}

at::Tensor rmsnorm(at::Tensor& output, const at::Tensor& x, 
    const c10::optional<at::Tensor>& weight, 
    int M, int N, double eps) {
  const int nthread = min(((N + 31) / 32) * 32, MAX_THREADS);
  dim3 grid_dim(M);
  dim3 block_dim(nthread);

  rmsnorm_kernel<<<grid_dim, block_dim>>>(
    reinterpret_cast<__half*>(x.data_ptr<at::Half>()),
    reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
    N
  );
  return output;
}

} // flash_ops
