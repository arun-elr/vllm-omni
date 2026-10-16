"""Read-only observation hooks for the vllm-omni (PR) side of the BWM parity run.

Active only when BWM_DUMP_DIR is set; put this directory on PYTHONPATH so the
diffusion worker subprocess picks it up too. Nothing here changes outputs:
each hook clones tensors to CPU and returns the original value.

* ``diffusers.utils.torch_utils.randn_tensor``: dumps every noise tensor the
  pipeline draws (``noise_w{k}.pt``), so the harness can assert it is
  bit-identical to the pre-generated noise file injected into the reference.
* ``WanEulerScheduler.step``: dumps the first DiT output of each window
  (``noise_pred0_w{k}.pt``) and the final denoised latents
  (``latents_w{k}.pt``).

Dumps go to ``$BWM_DUMP_DIR/<tag>/`` where ``<tag>`` is read from
``$BWM_DUMP_DIR/CURRENT`` (written by the driver before each rollout); the
window index ``k`` counts noise draws since the tag last changed.
"""

import importlib.abc
import importlib.machinery
import os
import sys

_DUMP = os.environ.get("BWM_DUMP_DIR")
_state = {"tag": None, "window": -1}


def _tag_dir():
    with open(os.path.join(_DUMP, "CURRENT")) as f:
        tag = f.read().strip()
    if tag != _state["tag"]:
        _state["tag"] = tag
        _state["window"] = -1
    d = os.path.join(_DUMP, tag)
    os.makedirs(d, exist_ok=True)
    return d


def _patch_torch_utils(mod):
    import torch

    orig = mod.randn_tensor

    def randn_tensor(*args, **kwargs):
        out = orig(*args, **kwargs)
        d = _tag_dir()
        _state["window"] += 1
        torch.save(out.detach().cpu().clone(), os.path.join(d, f"noise_w{_state['window']}.pt"))
        return out

    mod.randn_tensor = randn_tensor


def _patch_scheduler(mod):
    import torch

    cls = mod.WanEulerScheduler
    orig = cls.step

    def step(self, model_output, timestep, sample, *args, **kwargs):
        first = self.step_index is None
        out = orig(self, model_output, timestep, sample, *args, **kwargs)
        d = _tag_dir()
        k = _state["window"]
        if first:
            torch.save(model_output.detach().cpu().clone(), os.path.join(d, f"noise_pred0_w{k}.pt"))
        if self.step_index == len(self.timesteps):
            prev = out[0] if isinstance(out, tuple) else out.prev_sample
            torch.save(prev.detach().cpu().clone(), os.path.join(d, f"latents_w{k}.pt"))
        return out

    cls.step = step


_TARGETS = {
    "diffusers.utils.torch_utils": _patch_torch_utils,
    "vllm_omni.diffusion.models.wan2_2.scheduling_wan_euler": _patch_scheduler,
}


class _Loader(importlib.abc.Loader):
    def __init__(self, inner, patch):
        self.inner, self.patch = inner, patch

    def create_module(self, spec):
        return self.inner.create_module(spec)

    def exec_module(self, module):
        self.inner.exec_module(module)
        self.patch(module)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        patch = _TARGETS.get(name)
        if patch is None:
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(name, path, target)
            if spec is not None and spec.loader is not None:
                spec.loader = _Loader(spec.loader, patch)
                return spec
        return None


if _DUMP:
    sys.meta_path.insert(0, _Finder())
