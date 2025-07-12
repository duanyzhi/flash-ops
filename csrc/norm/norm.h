#pragma once
#include <torch/library.h>

namespace flash_ops {

at::Tensor layernorm_float(at::Tensor& output, const at::Tensor& x,
    const c10::optional<at::Tensor>& weight, const c10::optional<at::Tensor>& bias,
    int M, int N, double eps=1e-5);

}  // flash_ops
