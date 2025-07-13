#pragma once
#include <torch/library.h>

namespace flash_ops {

at::Tensor mma_forward(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias={});

}  /// flash ops
