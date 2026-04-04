import torch
from flash_ops import _C


a = torch.randn([32, 128]).float().cuda()

print(a, a.max())

torch_out = torch.nn.functional.softmax(a)

flash_out = _C.softmax(a)

print(a, torch_out, flash_out)

torch.testing.assert_close(torch_out, flash_out, rtol=0.1, atol=0.1)


