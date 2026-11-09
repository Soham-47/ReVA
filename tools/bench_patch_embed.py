#!/usr/bin/env python3
"""Time Qwen3-VL's Conv3d patch embed vs the equivalent linear, with cuDNN on and off (bf16, fwd+bwd)."""
import time

import torch
import torch.nn.functional as F

N = 25600  # ~32 frames at video_max_pixels=409600: 16 temporal groups x ~1600 patches
x = torch.randn(N, 3, 2, 16, 16, device="cuda", dtype=torch.bfloat16)
w = torch.randn(1152, 3, 2, 16, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True)
b = torch.zeros(1152, device="cuda", dtype=torch.bfloat16, requires_grad=True)


def conv():
    return F.conv3d(x, w, b, stride=(2, 16, 16)).view(N, -1)


def linear():
    return F.linear(x.view(N, -1), w.view(1152, -1), b)


def timeit(fn, reps=5):
    fn().sum().backward(); torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        fn().float().sum().backward()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps


print("gpu", torch.cuda.get_device_name(0), "torch", torch.__version__, "cudnn", torch.backends.cudnn.version())
print("max abs diff conv vs linear", (conv().float() - linear().float()).abs().max().item())
for cudnn in (True, False):
    torch.backends.cudnn.enabled = cudnn
    print(f"cudnn={cudnn} linear {timeit(linear)*1e3:.1f} ms", flush=True)
    print(f"cudnn={cudnn} conv3d {timeit(conv, reps=1)*1e3:.1f} ms", flush=True)
