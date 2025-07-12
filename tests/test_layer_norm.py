from flash_ops.layer_norm import LayerNorm
import torch
import torch.nn as nn
import pytest

# input shapes
SHAPES = [
    ## input shepe: (M, N) and norm shape: (N)
    # ((1, 1024), (1024)),
    ((1, 4096), (4096)),
    # ((1, 4096 * 4), (4096 * 4)),
    ((1, 4096 * 20), (4096 * 20)),
    # ((50, 4096), (4096)),
    ((4096, 4096), (4096)),
    ((2, 2304, 1280), (2304, 1280)), # batch, sentence_length, embedding_dim
     ((2, 2304, 1280), (1280)), 
    ((2, 9216 * 640), (9216 * 640)),
    ((20, 5, 10, 10), (10, 10)),  # N, C, H, W
    ((2, 5056, 640), (640)),
    ((2, 1280, 1280), (1280)),
]

DTYPES = [torch.float32, torch.float16, torch.bfloat16]
EPS = [1e-5, 1e-6]
BIAS = [True, False]
BACKEND = ["cuda"]
EA = [True, False]

# change iter for benchmark
WARE_UP = 1
ITER = 1

@pytest.mark.parametrize("Shape", SHAPES)
@pytest.mark.parametrize("Dtype", DTYPES)
@pytest.mark.parametrize("Eps", EPS)
@pytest.mark.parametrize("Bias", BIAS)
@pytest.mark.parametrize("Device", BACKEND)
@pytest.mark.parametrize("elementwise_affine", EA)
def test_layernorm(
    Shape,
    Dtype,
    Eps,
    Bias,
    Device,
    elementwise_affine,
):
    print("Benchmark for ", Shape)

    input_shape = Shape[0]
    normalized_shape = Shape[1]
    print("normalized_shape: ", normalized_shape)

    x = torch.randn(input_shape, device=Device, dtype=Dtype)

    norm = LayerNorm(normalized_shape=normalized_shape).to(Dtype).cuda()
    print(norm)

    base_out = None
    # ---------------------------------
    # pytorch native
    for _ in range(WARE_UP):
        # torch.nn.functional.layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-05)
        torch.nn.functional.layer_norm(x, norm.normalized_shape, weight=norm.weight, bias=norm.bias, eps=1e-05)
        
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for i in range(ITER):
        torch_out = torch.nn.functional.layer_norm(x, norm.normalized_shape, weight=norm.weight, bias=norm.bias, eps=1e-05)

    end.record() 
    torch.cuda.synchronize()
    print("avg pytorch layernorm sync time: ", start.elapsed_time(end) / ITER, " ms.")

    # ---------------------------------
    flash_out = None
    for _ in range(WARE_UP):
        norm(x)

    start_flash = torch.cuda.Event(enable_timing=True)
    end_flash = torch.cuda.Event(enable_timing=True)
    start_flash.record()   
    for i in range(ITER):
        flash_out = norm(x)

    end_flash.record() 
    torch.cuda.synchronize()
    print("avg flash fusion layernorm sync time: ", start_flash.elapsed_time(end_flash) / ITER, " ms.")

    # print(torch.allclose(base_out, flash_out, atol=1e-2, rtol=1e-2))

    max_bias = (abs(torch_out - flash_out)).max()
    print("max bias: ", max_bias)

    assert torch.allclose(torch_out, flash_out, atol=1e-2, rtol=1e-2) == True

