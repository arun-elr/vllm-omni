"""Read-only peak-memory logging for the BWM diffusion worker.

Active only when BWM_MEM_LOG is set; put this directory on PYTHONPATH so the
worker subprocess loads it. After each ``BoundlessWorldModelPipeline.forward``
it appends one JSON line with the process-lifetime peaks of
``torch.cuda.max_memory_allocated`` and ``torch.cuda.max_memory_reserved``
(never reset, so the last line is the peak over model load and all windows)
and the request's latent shape. Outputs are unchanged.
"""

import importlib.abc
import json
import os
import sys
import time

_LOG = os.environ.get("BWM_MEM_LOG")
_TARGET = "vllm_omni.diffusion.models.bwm.pipeline_bwm"


def _patch(mod):
    import torch

    cls = mod.BoundlessWorldModelPipeline
    orig = cls.forward

    def forward(self, req):
        t0 = time.perf_counter()
        out = orig(self, req)
        torch.cuda.synchronize()
        sp = req.sampling_params
        rec = {
            "time": time.time(),
            "seconds": round(time.perf_counter() - t0, 3),
            "height": sp.height,
            "width": sp.width,
            "num_frames": sp.num_frames,
            "dtype": str(self.transformer.dtype),
            "max_memory_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
            "max_memory_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 3),
        }
        with open(_LOG, "a") as f:
            f.write(json.dumps(rec) + "\n")
        return out

    cls.forward = forward


class _Loader(importlib.abc.Loader):
    def __init__(self, inner):
        self.inner = inner

    def create_module(self, spec):
        return self.inner.create_module(spec)

    def exec_module(self, module):
        self.inner.exec_module(module)
        _patch(module)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != _TARGET:
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(name, path, target)
            if spec is not None and spec.loader is not None:
                spec.loader = _Loader(spec.loader)
                return spec
        return None


if _LOG:
    sys.meta_path.insert(0, _Finder())
