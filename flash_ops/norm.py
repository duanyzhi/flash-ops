from typing import Union, List, Tuple

import torch
import torch.nn as nn
from torch import Tensor, Size
from flash_ops import _C

__all__ = ['LayerNorm']

_shape_t = Union[int, List[int], Size]

class LayerNorm(nn.LayerNorm):  # speedup for torch LayerNorm
    def __init__(self, normalized_shape: _shape_t, eps: float = 1e-5, elementwise_affine: bool = True,
                 bias: bool = True, device=None, dtype=None) -> None:
        super().__init__(normalized_shape, eps, elementwise_affine, bias, device, dtype)   # init torhc Layernorm
        pass

    def forward(self, input: Tensor) -> Tensor:
        return _C.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)

class RMSNorm(nn.RMSNorm): 
    def __init__(self, normalized_shape: _shape_t, eps: float = 1e-5, elementwise_affine: bool = True,
                device=None, dtype=None) -> None:
        super().__init__(normalized_shape, eps, elementwise_affine, device, dtype)   # init torhc Layernorm
        pass

    def forward(self, input: Tensor) -> Tensor:
        return _C.rms_norm(input, self.normalized_shape, self.weight, self.eps)
