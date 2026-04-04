#pragma once
#include <torch/library.h>

namespace flash_ops {

at::Tensor mma_forward(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias={});

at::Tensor multi_stage_mma_forward(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias={});

at::Tensor simple_gemm(const at::Tensor & A, const at::Tensor & B, const c10::optional<at::Tensor> & bias={});

}  /// flash ops
