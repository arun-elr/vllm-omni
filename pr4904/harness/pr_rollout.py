"""PR side: run examples/offline_inference/bwm/bwm_world_model.py main() unchanged.

For each variant tag this sets sys.argv and calls the example's ``main()``.
The driver only observes or wraps from outside:

* one ``Omni`` instance is reused across variants (constructed with the
  example's own kwargs plus an optional ``dtype``);
* ``imageio.get_writer`` is wrapped to keep the exact uint8 frames the
  example writes (pre-encoding), next to the mp4;
* ``load_episode`` is wrapped to save the first frame and normalized actions;
* ``<tag>:nocarry`` (negative control) replaces the history frames of
  windows >= 1 with copies of frame 0, i.e. breaks history carry-over.

Initial noise is the pipeline's own seeded draw (seed + window); the
sitecustomize hook dumps it for the bit-equality check against the saved
noise files.
"""

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--example", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--episode-dir", required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--tags", nargs="+", required=True)
    p.add_argument("--dtype", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=50)
    args = p.parse_args()

    spec = importlib.util.spec_from_file_location("bwm_world_model", args.example)
    ex = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ex)

    import vllm_omni
    import vllm_omni.entrypoints.omni as omni_mod

    print(f"vllm_omni imported from {Path(vllm_omni.__file__).parent}", flush=True)

    real_omni = omni_mod.Omni
    cache = {}
    ctx = {"tag": None, "nocarry": False, "window": 0, "log": []}

    def cached_omni(**kwargs):
        if "inst" not in cache:
            if args.dtype:
                kwargs["dtype"] = args.dtype
            cache["kwargs"] = {k: str(v) for k, v in kwargs.items()}
            inst = real_omni(**kwargs)
            real_generate = inst.generate

            def generate(prompt, sampling_params, *a, **kw):
                w = ctx["window"]
                mm = prompt["multi_modal_data"]
                if ctx["nocarry"] and w >= 1:
                    mm["video"] = np.repeat(mm["video"][:1], mm["video"].shape[0], axis=0)
                t0 = time.perf_counter()
                out = real_generate(prompt, sampling_params, *a, **kw)
                ctx["log"].append(
                    {
                        "window": w,
                        "history_shape": list(mm["video"].shape),
                        "action_shape": list(np.asarray(mm["action"]).shape),
                        "num_frames": sampling_params.num_frames,
                        "seed": sampling_params.seed,
                        "height": sampling_params.height,
                        "width": sampling_params.width,
                        "steps": sampling_params.num_inference_steps,
                        "seconds": round(time.perf_counter() - t0, 3),
                    }
                )
                ctx["window"] += 1
                return out

            inst.generate = generate
            cache["inst"] = inst
        return cache["inst"]

    omni_mod.Omni = cached_omni

    real_load = ex.load_episode

    def load_episode(*a, **kw):
        first_frame, action, meta = real_load(*a, **kw)
        np.savez(ctx["out"] / "inputs.npz", first_frame=first_frame, action=action)
        return first_frame, action, meta

    ex.load_episode = load_episode

    real_writer = ex.imageio.get_writer

    class Capture:
        def __init__(self, path, **kw):
            self.inner = real_writer(path, **kw)
            self.frames = []

        def append_data(self, frame):
            self.frames.append(np.asarray(frame))
            self.inner.append_data(frame)

        def close(self):
            self.inner.close()
            np.save(ctx["out"] / "frames_u8.npy", np.stack(self.frames))

    ex.imageio.get_writer = Capture

    dump = os.environ.get("BWM_DUMP_DIR")
    for spec_tag in args.tags:
        tag, _, mode = spec_tag.partition(":")
        out = args.out_dir / tag
        out.mkdir(parents=True, exist_ok=True)
        ctx.update(tag=tag, nocarry=(mode == "nocarry"), window=0, log=[], out=out)
        if dump:
            Path(dump, "CURRENT").write_text(tag)
        sys.argv = [
            "bwm_world_model.py",
            "--model", args.model,
            "--episode-dir", args.episode_dir,
            "--episode", "0",
            "--output", str(out / "rollout.mp4"),
            "--num-inference-steps", str(args.steps),
            "--seed", str(args.seed),
            "--enforce-eager",
        ]
        print(f"=== PR variant {spec_tag}: {' '.join(sys.argv)}", flush=True)
        t0 = time.perf_counter()
        ex.main()
        meta = {
            "variant": spec_tag,
            "argv": sys.argv,
            "omni_kwargs": cache.get("kwargs"),
            "vllm_omni_package": str(Path(vllm_omni.__file__).parent),
            "windows": ctx["log"],
            "rollout_seconds": round(time.perf_counter() - t0, 2),
        }
        (out / "run.json").write_text(json.dumps(meta, indent=2))
        print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
