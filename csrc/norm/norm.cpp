#include "../ops.h"
#include "norm.h"
#include <torch/torch.h>

namespace flash_ops {

at::Tensor layer_norm(const at::Tensor x, at::IntArrayRef normalized_shape, const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& bias, double eps) {

    const auto input_shape = x.sizes();
    const size_t axis = x.dim() - normalized_shape.size();

    int M = 1;
    int norm_size = 1;
    for (size_t idx = axis; idx < input_shape.size(); ++idx) {
        norm_size *= x.size(idx);
    }
    for (const auto idx: c10::irange(axis)) {
        M *= x.size(idx);
    }
    auto output = torch::empty_like(x);

    if (x.scalar_type() == c10::ScalarType::Float) {
        return layernorm_float(output, x, weight, bias, M, norm_size, eps);
    }
    return at::layer_norm(x, normalized_shape, weight, bias);
}

at::Tensor rms_norm(const at::Tensor x, at::IntArrayRef normalized_shape, const c10::optional<at::Tensor>& weight, double eps) {

    const auto input_shape = x.sizes();
    const size_t axis = x.dim() - normalized_shape.size();

    int M = 1;
    int norm_size = 1;
    for (size_t idx = axis; idx < input_shape.size(); ++idx) {
        norm_size *= x.size(idx);
    }
    for (const auto idx: c10::irange(axis)) {
        M *= x.size(idx);
    }
    auto output = torch::empty_like(x);

    if (x.scalar_type() == c10::ScalarType::Half) {
        return rmsnorm(output, x, weight, M, norm_size, eps);
    }
    return at::rms_norm(x, normalized_shape, weight, eps);
}

}  // namespace flash_fusion

