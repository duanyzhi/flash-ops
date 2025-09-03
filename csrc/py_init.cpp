#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/torch.h>
#include <iostream>
#include "ops.h"

namespace py = pybind11;

PYBIND11_MODULE(_C, m) {
    m.def("init", []() { std::cout << "flash ops init" << std::endl; });
    // base kernel
    m.def("layer_norm", &flash_ops::layer_norm, "Flash LayerNorm Op.");
    m.def("rms_norm", &flash_ops::rms_norm, "Flash RMSNorm Op.");
    m.def("linear", &flash_ops::linear, "Flash Linear Op.");
}

