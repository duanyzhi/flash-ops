#pragma once

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace flash_ops {

const unsigned int WARP_REDUCE_MASK = 0xffffffff;
#define MAX_THREADS 1024  // Max threads per block for Ada
#define DIV_UP(x, y) ((x) + (y) - 1) / (y)
#define WARP_SIZE 32
#define OFFSET(row, col, ld) ((row) * (ld) + (col))

__device__ __forceinline__ float convert_fp16_to_fp32(half data) { return __half2float(data); }
__device__ __forceinline__ float convert_fp16_to_fp32(nv_bfloat16 data) { return __bfloat162float(data); }

template <typename T>
__device__ __forceinline__ T convert_fp32_to_fp16(float data) {
    return;
}

template <>
__device__ __forceinline__ half convert_fp32_to_fp16<half>(float data) {
    return __float2half(data);
}

template <>
__device__ __forceinline__ nv_bfloat16 convert_fp32_to_fp16<nv_bfloat16>(float data) {
    return __float2bfloat16(data);
}

inline __device__ float warp_reduce_sum(float val) {
  val += __shfl_xor_sync(~0, val, 16);
  val += __shfl_xor_sync(~0, val, 8);
  val += __shfl_xor_sync(~0, val, 4);
  val += __shfl_xor_sync(~0, val, 2);
  val += __shfl_xor_sync(~0, val, 1);
  return val;
}

template<typename T, int NUM>
__inline__ __device__ T warpReduceSum(T* val)
{
#pragma unroll
    for (int i = 0; i < NUM; i++) {
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1)
            val[i] += __shfl_xor_sync(WARP_REDUCE_MASK, val[i], mask, 32);
    }
    return (T)(0.0f);
}

__device__ __forceinline__ float blockReduceSum(float reducing, float *shared_mem) {
    // Helper function for reduce softmax exp sum.
    const int32_t WPT = blockDim.x / 32;
    int32_t WPTB = WPT < 32 ? 32 : WPT;
    const int32_t lane_id = threadIdx.x % 32;
    const int32_t warp_id = threadIdx.x / 32;

#pragma unroll
    for (int32_t mask = 16; mask >= 1; mask /= 2) {
        reducing += __shfl_xor_sync(uint32_t(-1), reducing, mask);
    }

    if (lane_id == 0) shared_mem[warp_id] = reducing;
    __syncthreads();

    if (lane_id < WPTB) reducing = lane_id < WPT ? shared_mem[lane_id] : 0.0f;

#pragma unroll
    for (int32_t mask = WPTB / 2; mask >= 1; mask /= 2) {
        reducing += __shfl_xor_sync(uint32_t(-1), reducing, mask);
    }
    reducing = __shfl_sync(uint32_t(-1), reducing, 0);
    return reducing;
}

}  // namespace flash_ops
