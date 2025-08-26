#include "../ops.h"
#include "gemm.h"

namespace flash_ops {

at::Tensor linear(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias) {
  const bool on_device = input.device().type() == c10::DeviceType::CUDA && \
		     weight.device().type() == c10::DeviceType::CUDA;
  TORCH_CHECK(on_device, "input and weight must be CUDA device.");
  if (input.scalar_type() == at::kHalf) {
    return multi_stage_mma_forward(input, weight, bias);
  }
  return at::linear(input, weight, bias);
}

}  // namespace flash_ops
