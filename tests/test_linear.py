import torch
from flash_ops import _C
import pytest
import nvtx
from tabulate import tabulate

torch.manual_seed(12138)
torch.cuda.manual_seed_all(12138) 
# torch.set_printoptions(sci_mode=False, threshold=float('inf'))
# torch.set_printoptions(sci_mode=False)

TestInfos = [
    # shape, dtype, bias, device, atol, speedup
    # ((1, 1024, 1024 + 32 + 32, 1024), torch.half, False, "cuda", 0.01, 0.01),
    ((1, 2048, 2048, 2048), torch.half, False, "cuda", 0.04, 0.2),
    # ((1, 2048, 2048, 2048), torch.half, False, "cuda", 0.1, 0.1),
    # ((1, 128, 128, 128), torch.half, False, "cuda", 0.04, 0.2),
    # ((1, 8192, 8192, 8192), torch.half, True, "cuda", 0.04, 0.2),
    # ((1, 11264, 11264, 11264), torch.half, True, "cuda", 0.04, 0.2),
    # ((1, 16384, 16384, 16384), torch.half, True, "cuda", 0.04, 0.2),
]

class Linear(torch.nn.Module):
    def __init__(self, in_feature, out_feature, bias, dtype, device):
        super().__init__()
        self.ln = torch.nn.Linear(in_feature, out_feature, bias=bias, device=device, dtype=dtype)

    def forward(self, x):
        return self.ln(x)
    
iter = 100

RESULTS = []

 #"(M, K, N)", "sync time/ms", "(Throughput/GFLOPS)"],


@pytest.mark.parametrize("Infos", TestInfos)
def test_linear(
    Infos
):
  Shape, Dtype, Bias, Device, Atol, Speedup = Infos
  #print("=="*30)
  #print(Infos)
  line = []
  B, M, K, N = Shape
  line.append((M, K, N))
  
  x = torch.randn([B, M, K], device=Device, dtype=Dtype)
  
  layer = Linear(K, N, Bias, Dtype, Device)
  # print("x: 0", x[0, 0:16, 0:16])
  # print("w: 0", layer.ln.weight[0:32, 0:32])
#   print("x: 1", x[0, 0:16, 16:32])
#   print("w: 1", layer.ln.weight[0:32, 0:32])
#   print("x0 @ w0: ", x[0, 0, 0:16] @ layer.ln.weight[2, 0:16].t())
#   print("x1 @ w1: ", x[0, 0, 16:32] @ layer.ln.weight[2, 16:32].t())
  for _ in range(10):
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
  #print("avg pytorch linear sync time: ", start.elapsed_time(end) / iter, " ms.")

  compute_flops = M * (K + (K - 1)) * N
  #print("FLOPS: ", compute_flops)
  torch_time = start.elapsed_time(end) / iter

  torch_throughput = compute_flops / (torch_time / 1000) / (10 ** 9)
  #print("Torch Throughput: ", torch_throughput, " GFLOPS")
  #print(x)
  print("-------------------------------------------")
  
  with nvtx.annotate('flash_ops'):
      start_flash = torch.cuda.Event(enable_timing=True)
      end_flash = torch.cuda.Event(enable_timing=True)
      start_flash.record()
      for _ in range(iter):
        flash_out = _C.linear(x, layer.ln.weight, layer.ln.bias)

      end_flash.record() 
      torch.cuda.synchronize()
  #print("avg flash linear sync time: ", start_flash.elapsed_time(end_flash) / iter, " ms.")

  flash_time = start_flash.elapsed_time(end_flash) / iter

  flash_throughput = compute_flops / (flash_time / 1000) / (10 ** 9)
#   print("o0: ", torch_out.size(), torch_out[0, 0, 0:16], flash_out[0, 0, 0:16])
#   print("o1: ", torch_out[0, 0, 16:32])
  #print("Flash Throughput: ", flash_throughput, " GFLOPS")
  # print("torch: ", torch_out, "\nflash:", flash_out)
  # print(torch_out[0, 48, 17], flash_out[0, 48, 17])

  # for r in range(0, torch_out.size(1)):
  #   try:
  #     z = torch.testing.assert_close(flash_out[0, r, :].float(), torch_out[0, r, :].float(), rtol=0, atol=0.5)
  #     # print(z)
  #   except AssertionError as e:
  #     # if (r == 115):
  #     if z is None:
  #       # print("torch and flash: ", r, torch_out[0, r, :], flash_out[0, r, :])
  #       print(torch_out[0, r, :] - flash_out[0, r, :])
  #       abs_diff = torch.abs(torch_out[0, r, :] - flash_out[0, r, :])
  #       max_abs_diff, max_idx = torch.max(abs_diff, dim=0)
  #       max_abs_diff = max_abs_diff.item()
  #       max_idx = max_idx.item()
  #       print("row and col: ", r, max_idx / 32)
  #       print("max_abs_diff: ", max_abs_diff, max_idx, torch_out[0, r, max_idx], flash_out[0, r, max_idx])

  print(torch_out, flash_out)
  # torch.testing.assert_close(torch_out.float(), flash_out.float(), rtol=0, atol=0.5)
  # atol: abs(actual - expected)

  result = {
        "Shape": f"({M}, {K}, {N})",
        "PyTorch Time (ms)": f"{torch_time:.2f}",
        "Flash Time (ms)": f"{flash_time:.2f}",
        "PyTorch GFLOPS": f"{torch_throughput:.2f}",
        "Flash GFLOPS": f"{flash_throughput:.2f}",
        "Speedup": f"{torch_time/flash_time:.2f}x"
  }
  RESULTS.append(result)


@pytest.fixture(scope="session", autouse=True)
def print_results():
    yield
    if RESULTS:
        #headers = RESULTS[0].keys()
        #print(headers)
        #rows = [list(result.values()) for result in RESULTS]
        #print(rows)
        # print(tabulate(rows, headers=headers, tablefmt="grid", stralign="right", numalign="right"))
        print(RESULTS)
        print(tabulate(RESULTS, headers="keys", tablefmt="fancy_grid"))


