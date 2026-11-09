#!/usr/bin/env python3
"""Prove the Hopper cuModuleLoadData failure is an empty cached NVRTC kernel, and that the fix holds.

Runs the exact op from transformers Qwen3-VL get_image_features
(`(grid_thw.prod(-1) // merge**2).tolist()` on an int64 CUDA tensor) in fresh processes:

  A reproduce: cache on, a 0-byte file planted under the real cache entry name -> must fail
  B fix:       USE_PYTORCH_KERNEL_CACHE=0 (what jobs/activate_reva.sh sets)  -> must pass
  C sanity:    cache on, empty directory -> must pass and write a non-empty kernel file

Exit 0 only if B and C pass. A is reported (REPRODUCED / NOT_REPRODUCED).
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

OP = (
    "import torch;"
    "g=torch.tensor([[8,22,40],[16,22,40]],device='cuda');"
    "print('result', (g.prod(-1)//4).tolist(), torch.cuda.get_device_name(0), flush=True)"
)
# Name of the entry that was 0 bytes in /home/ambpdc/.cache/torch/kernels (sm_90, nvrtc 13.0).
EMPTY_ENTRY = "reduction_prod_arch9.0_nvrtc13.0_sass_33436_e4ad3d09bf929701ca4a6d9e3736737e87b3e7a5"


def run(name: str, env_updates: dict[str, str]) -> tuple[bool, str]:
    env = dict(os.environ, **env_updates)
    proc = subprocess.run([sys.executable, "-c", OP], env=env, capture_output=True, text=True, timeout=300)
    out = (proc.stdout + proc.stderr).strip()
    print(f"--- {name} rc={proc.returncode}\n{out[-1500:]}", flush=True)
    return proc.returncode == 0, out


def main() -> int:
    import torch

    major, minor = torch.cuda.get_device_capability(0)
    print(f"gpu={torch.cuda.get_device_name(0)} sm_{major}{minor} torch={torch.__version__} "
          f"nvrtc_cuda={torch.version.cuda}", flush=True)
    root = Path(tempfile.mkdtemp(prefix="reva_kcache_", dir=os.environ.get("TMPDIR", "/tmp")))

    a_dir = root / "A"; a_dir.mkdir()
    (a_dir / EMPTY_ENTRY).touch()
    a_ok, a_out = run("A reproduce (empty cached kernel)", {
        "USE_PYTORCH_KERNEL_CACHE": "1", "PYTORCH_KERNEL_CACHE_PATH": str(a_dir)})
    if (major, minor) != (9, 0):
        a_verdict = "SKIPPED_NOT_SM90"
    elif not a_ok and "image cannot be NULL" in a_out:
        a_verdict = "REPRODUCED"
    else:
        a_verdict = "NOT_REPRODUCED"

    b_ok, _ = run("B fix (USE_PYTORCH_KERNEL_CACHE=0)", {"USE_PYTORCH_KERNEL_CACHE": "0"})

    c_dir = root / "C"; c_dir.mkdir()
    c_ok, _ = run("C fresh cache", {"USE_PYTORCH_KERNEL_CACHE": "1", "PYTORCH_KERNEL_CACHE_PATH": str(c_dir)})
    written = sorted((p.name, p.stat().st_size) for p in c_dir.iterdir())
    c_ok = c_ok and bool(written) and all(size > 0 for _, size in written)
    print(f"C wrote: {written}", flush=True)

    print(f"VERDICT A={a_verdict} B={'PASS' if b_ok else 'FAIL'} C={'PASS' if c_ok else 'FAIL'}", flush=True)
    return 0 if (b_ok and c_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
