import torch
from flash_ops import _C
import pytest
import nvtx

torch.manual_seed(12138)
torch.cuda.manual_seed_all(12138) 

TestInfos = [
    # shape, dtype, bias, device, atol, speedup
    ((1, 1024, 1024, 1024), torch.half, True, "cuda", 0.04, 0.2),
    ((1, 2048, 2048, 2048), torch.half, True, "cuda", 0.04, 0.2),
    ((1, 4096, 4096, 4096), torch.half, True, "cuda", 0.04, 0.2),
    ((1, 8192, 8192, 8192), torch.half, True, "cuda", 0.04, 0.2),
    ((1, 11264, 11264, 11264), torch.half, True, "cuda", 0.04, 0.2),
    ((1, 16384, 16384, 16384), torch.half, True, "cuda", 0.04, 0.2),
]

class Linear(torch.nn.Module):
    def __init__(self, in_feature, out_feature, bias, dtype, device):
        super().__init__()
        self.ln = torch.nn.Linear(in_feature, out_feature, bias=bias, device=device, dtype=dtype)

    def forward(self, x):
        return self.ln(x)
    
iter = 50

@pytest.mark.parametrize("Infos", TestInfos)
def test_linear(
    Infos
):
  Shape, Dtype, Bias, Device, Atol, Speedup = Infos
  print("=="*30)
  print(Infos)
  B, M, K, N = Shape
  x = torch.randn([B, M, K], device=Device, dtype=Dtype)
  
  layer = Linear(K, N, Bias, Dtype, Device)
  
  for _ in range(5):
     layer(x)
     _C.linear(x, layer.ln.weight, None)

  torch.cuda.synchronize()

  with nvtx.annotate('torch_linear'):
      start = torch.cuda.Event(enable_timing=True)
      end = torch.cuda.Event(enable_timing=True)
      start.record()
      for _ in range(iter):
          torch_out = layer(x)

      end.record() 
      torch.cuda.synchronize()
  print("avg pytorch linear sync time: ", start.elapsed_time(end) / iter, " ms.")

  compute_flops = M * (K + (K - 1)) * N
  print("FLOPS: ", compute_flops)
  torch_time = start.elapsed_time(end) / iter / 1000

  torch_throughput = compute_flops / torch_time / (10 ** 9)
  print("Torch Throughput: ", torch_throughput, " GFLOPS")
  
  with nvtx.annotate('flash_ops'):
      start_flash = torch.cuda.Event(enable_timing=True)
      end_flash = torch.cuda.Event(enable_timing=True)
      start_flash.record()
      for _ in range(iter):
        flash_out = _C.linear(x, layer.ln.weight, layer.ln.bias)

      end_flash.record() 
      torch.cuda.synchronize()
  print("avg flash linear sync time: ", start_flash.elapsed_time(end_flash) / iter, " ms.")

  flash_time = start_flash.elapsed_time(end_flash) / iter / 1000.0

  flash_throughput = compute_flops / flash_time / (10 ** 9)
  print("Flash Throughput: ", flash_throughput, " GFLOPS")
 
  torch.testing.assert_close(torch_out.float(), flash_out.float(), atol=Atol, rtol=Atol)
