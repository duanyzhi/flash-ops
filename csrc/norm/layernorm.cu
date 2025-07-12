#include "norm.h"
#include "cuda_utils.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/AccumulateType.h>
#include <iostream>
#include <cuda_fp16.h>

namespace flash_ops {

__global__ void layernorm_kernel(const float4* __restrict__ x, const float4* __restrict__ weight,
    const float4* __restrict__ bias, float eps, float4* __restrict__ output, int hidden_dim, bool enable_weight,
    bool enable_bias) {
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    const float4* x_block = x + blockIdx.x * hidden_dim;
    uint stride = blockDim.x;
    float sum_x = 0;
    float sum_xx = 0;
    float4 val_next = x_block[tid];

    #pragma unroll
    for (uint id = tid; id < hidden_dim; id+=stride) {
      float4 val = val_next;
      if (id + stride < hidden_dim) {
          val_next = x_block[id + stride];
      }

      float sum_val = val.x + val.y + val.z + val.w;
      float sum_sq_val =
          val.x * val.x + val.y * val.y + val.z * val.z + val.w * val.w;
      sum_x += sum_val;
      sum_xx += sum_sq_val;
    }

    float reduce_val[2] = {sum_x, sum_xx};
    warpReduceSum<float, 2>(reduce_val);

    int lane_id = threadIdx.x & 0x1f;
    int wid = threadIdx.x >> 5;

    static __shared__ float shared[2][32];
    if (lane_id == 0) {  // push all lane_id 0 to shared
      shared[0][wid] = reduce_val[0];
      shared[1][wid] = reduce_val[1];
    }

    __syncthreads();
    if (threadIdx.x < (blockDim.x >> 5)) {  // < 32
      reduce_val[0] = shared[0][lane_id];  // get mean from shared memory
      reduce_val[1] = shared[1][lane_id];  // get mean from shared memory
    } else {
      reduce_val[0] = 0.0f;  // only keey warp data
      reduce_val[1] = 0.0f;  // only keey warp data
    }
    warpReduceSum<float, 2>(reduce_val);

    __shared__ float reduce_mean, reduce_var;
    if (tid == 0) {
       reduce_mean = __ddiv_rn(reduce_val[0], hidden_dim * 4.0f);
       reduce_var = __ddiv_rn(reduce_val[1], hidden_dim * 4.0f) - __dmul_rn(reduce_mean, reduce_mean) + eps;
       reduce_var = rsqrtf(reduce_var);
    }
    __syncthreads(); // wait reduce_mean and var

    float4 *output_block = output + blockIdx.x * hidden_dim;

    #pragma unroll
    for (uint idx = tid; idx < hidden_dim; idx += blockDim.x) {
      float4 val = x_block[idx];
      val.x = (val.x - reduce_mean) * reduce_var;
      val.y = (val.y - reduce_mean) * reduce_var;
      val.z = (val.z - reduce_mean) * reduce_var;
      val.w = (val.w - reduce_mean) * reduce_var;

      if (enable_weight) {
        float4 vscale = __ldg(reinterpret_cast<const float4 *>(weight) + idx);
        val.x = val.x * vscale.x;
        val.y = val.y * vscale.y;
        val.z = val.z * vscale.z;
        val.w = val.w * vscale.w;
      }
      if (enable_bias) {
        float4 vbias = __ldg(reinterpret_cast<const float4 *>(bias) + idx);
        val.x = val.x + vbias.x;
        val.y = val.y + vbias.y;
        val.z = val.z + vbias.z;
        val.w = val.w + vbias.w;
      }
      output_block[idx] = val;
    }
}

at::Tensor layernorm_float(at::Tensor& output, const at::Tensor& x, 
    const c10::optional<at::Tensor>& weight, const c10::optional<at::Tensor>& bias,
    int M, int N, double eps) {
  const int batch_size = M;
  int hidden_dim = N;
  const int nthread = min(((hidden_dim + 31) / 32) * 32, MAX_THREADS);
  dim3 grid_dim(batch_size);
  dim3 block_dim(nthread);
  const float4* x_f4 = reinterpret_cast<const float4 *>(x.data_ptr());
  const float4* w_f4 = nullptr;
  const float4* b_f4 = nullptr;
  bool enable_weight = false;
  bool enable_bias = false;
  if (weight.has_value()) {
    w_f4 = reinterpret_cast<const float4 *>(weight.value().data_ptr());
    enable_weight = true;
  }
  if (bias.has_value()) {
    b_f4 = reinterpret_cast<const float4 *>(bias.value().data_ptr());
    enable_bias = true;
  }
  float4* o_f4 = reinterpret_cast<float4 *>(output.data_ptr());
  hidden_dim = hidden_dim / 4;
  layernorm_kernel<<<grid_dim, block_dim>>>(
      x_f4, w_f4, b_f4,
      static_cast<float>(eps),
      o_f4, hidden_dim,
      enable_weight, enable_bias);
  return output;
}

} // flash_ops
