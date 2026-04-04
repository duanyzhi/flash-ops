import math

import torch
from torch.nn import functional as F
# from torch.utils.cpp_extension import load
from flash_ops import _C

# Load the CUDA kernel as a python module
# minimal_attn = load(name='minimal_attn', sources=['main.cpp', 'flash.cu'], extra_cuda_cflags=['-O2'])

# Use small model params, otherwise slower than manual attention. See caveats in README.
batch_size = 1
n_head = 8
kv_head = 8
seq_len = 32
head_dim = 64 # 128

q = torch.randn(batch_size, n_head, seq_len, head_dim).cuda()
k = torch.randn(batch_size, kv_head, seq_len, head_dim).cuda()
v = torch.randn(batch_size, kv_head, seq_len, head_dim).cuda()

print('=== profiling manual attention ===')

scale = (1.0 / math.sqrt(head_dim))
kt = k.transpose(-2, -1)

# Our minimal flash attention aims to be faster than this by avoiding HBM read/writes of N^2 matrices.
def manual_attn(q, k, v):
    att = (q @ k * scale)

    att = F.softmax(att, dim=-1)
    y = att @ v
    return y

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    manual_result = manual_attn(q, kt, v)
# print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

# print('=== profiling minimal flash attention === ')

# QQ = torch.randn(32, 128).float().cuda()
# KK = torch.randn(128, 32).float().cuda()
o = _C.attention(q, kt, v)
print(manual_result[0, 0, ...], o[0, 0, ...])

torch.testing.assert_close(manual_result, o, rtol=0.1, atol=0.1)


# o = _C.attention(QQ, KK, v)

# torch_o = torch.matmul(QQ, KK)
# print(torch_o, o)

# with torch.autograd.profiler.profile(use_cuda=True) as prof:
#     minimal_result = minimal_attn.forward(q, k, v)
# print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

# print('attn values sanity check:', torch.allclose(minimal_result, manual_result, rtol=0, atol=1e-02))
