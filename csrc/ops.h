#pragma once

#include <torch/python.h>
#include <torch/nn/functional.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <c10/core/DispatchKey.h>
#include <c10/macros/Macros.h>
#include <c10/util/MaybeOwned.h>
#include <c10/util/irange.h>
#include <torch/library.h>

namespace flash_ops {

at::Tensor layer_norm(const at::Tensor x, at::IntArrayRef normalized_shape, const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& bias, double eps = 1e-5);

at::Tensor rms_norm(const at::Tensor x, at::IntArrayRef normalized_shape, const c10::optional<at::Tensor>& weight,
    double eps = 1e-5);

at::Tensor linear(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias={});

}  // namespace flash_ops
