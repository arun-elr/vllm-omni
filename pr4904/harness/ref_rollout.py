"""Reference side: the BWM repo's scripts/infer.py rollout with injected noise.

Uses the reference's own argument parsing (configs/infer/infer.yaml plus CLI
overrides for paths and resolution), dataset loader, ``build_pipeline`` and
``_run_autoregressive``, unmodified. Hooks, all applied from outside:

* noise (the only change on the default path): ``pipe.generate_noise`` returns
  the saved tensor for the current window instead of a CPU-generator draw, cast
  exactly as ``generate_noise`` casts (to the pipeline dtype and device). It
  asserts the shape and the seed the reference asked for (seed + window).
* observation: the final latents (VAE decode input) and the first DiT output
  of each window are dumped; the rollout's frames are kept as uint8 using the
  reference's own ``_normalize_to_uint8``.
* diagnostics, opt-in per variant: ``:nohistnoise`` passes
  ``use_history_condition_noise_in_inference=False`` (infer.py hardcodes True);
  ``:nocarry`` replaces the history frames of windows >= 1 with copies of
  frame 0; ``:quantcarry`` carries generated history frames as truncated
  uint8, the way the PR example does; ``:noise1042`` injects the seed-1042
  noise (negative control).
* ``--dtype fp32`` rebuilds the pipeline with ``torch.float32``
  (``build_pipeline`` only offers bf16/fp16); otherwise ``build_pipeline`` is
  used as-is.
"""

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ref-repo", type=Path, required=True)
    p.add_argument("--model-paths", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--noise-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--tags", nargs="+", required=True)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=640)
    args = p.parse_args()

    out_root = args.out_dir.resolve()
    noise_dir = args.noise_dir.resolve()
    os.chdir(args.ref_repo)
    sys.path.insert(0, str(args.ref_repo.resolve()))
    spec = importlib.util.spec_from_file_location("ref_infer", args.ref_repo / "scripts" / "infer.py")
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    from wan_video_action.utils import _normalize_to_uint8

    sys.argv = [
        "infer.py",
        "--config", "configs/infer/infer.yaml",
        "--model_paths", args.model_paths,
        "--ckpt_path", args.ckpt,
        "--dataset_base_path", "demo",
        "--dataset_metadata_path", "demo/demo.jsonl",
        "--action_stat_path", "demo/stat.json",
        "--output_path", str(out_root / "_ref_out"),
        "--height", str(args.height),
        "--width", str(args.width),
        "--max_samples", "1",
    ]
    rargs = ref.parse_args()
    resolved = {k: str(v) for k, v in sorted(vars(rargs).items())}
    print("[resolved_args]", json.dumps(resolved), flush=True)

    dataset = ref.build_infer_dataset(rargs)
    if args.dtype == "bf16":
        pipe = ref.build_pipeline(rargs)
    else:
        rc = ref.prepare_runtime_config(rargs)
        model_configs = [ref.ModelConfig(path=ref.resolve_model_path(m)) for m in rc["model_paths_list"]]
        pipe = ref.build_wan_video_action_pipeline(
            torch_dtype=torch.float32,
            device="cuda",
            model_configs=model_configs,
            tokenizer_config=None,
            ckpt_path=rargs.ckpt_path,
            action_dim=rargs.action_dim,
            action_mode=rargs.action_mode,
        )
        pipe.use_gradient_checkpointing = False
        pipe.use_gradient_checkpointing_offload = False
        pipe.eval()

    ctx = {}

    def generate_noise(shape, seed=None, rand_device="cpu", rand_torch_dtype=torch.float32, device=None, torch_dtype=None):
        w = ctx["window"]
        assert seed == rargs.seed + w, (seed, rargs.seed, w)
        f = noise_dir / f"seed{ctx['noise_seed']}_w{w}.pt"
        noise = torch.load(f)
        assert tuple(noise.shape) == tuple(shape), (tuple(noise.shape), tuple(shape))
        ctx["noise_files"].append(f.name)
        return noise.to(dtype=torch_dtype or pipe.torch_dtype, device=device or pipe.device)

    pipe.generate_noise = generate_noise

    real_decode = pipe.vae.decode

    def decode(latents, *a, **kw):
        torch.save(latents.detach().cpu().clone(), ctx["out"] / f"latents_w{ctx['window']}.pt")
        return real_decode(latents, *a, **kw)

    pipe.vae.decode = decode

    real_step = pipe.scheduler.step

    def step(model_output, timestep, sample, *a, **kw):
        if ctx["steps_in_window"] == 0:
            torch.save(model_output.detach().cpu().clone(), ctx["out"] / f"noise_pred0_w{ctx['window']}.pt")
        ctx["steps_in_window"] += 1
        return real_step(model_output, timestep, sample, *a, **kw)

    pipe.scheduler.step = step

    cls = type(pipe)
    real_call = cls.__call__

    def call(self, *a, **kw):
        w = ctx["window"]
        if ctx["histnoise"] is not None:
            kw["use_history_condition_noise_in_inference"] = ctx["histnoise"]
        if ctx["nocarry"] and w >= 1:
            iv = kw["input_video"]
            kw["input_video"] = iv[:, :, :1].expand_as(iv).clone()
        if ctx["quantcarry"] and w >= 1:
            # The PR example carries generated frames as uint8, truncating
            # (x / 2 + 0.5).clamp(0, 1) * 255; the reference carries floats.
            # Frame 0 (history slot 0) is the dataset frame on both sides.
            iv = kw["input_video"].clone()
            gen = iv[:, :, 1:].float()
            u8 = torch.trunc((gen / 2 + 0.5).clamp(0, 1) * 255)
            iv[:, :, 1:] = (u8 / 127.5 - 1.0).to(iv.dtype)
            kw["input_video"] = iv
        ctx["steps_in_window"] = 0
        t0 = time.perf_counter()
        out = real_call(self, *a, **kw)
        ctx["log"].append(
            {
                "window": w,
                "history_shape": list(kw["input_video"].shape),
                "action_shape": list(kw["action"].shape),
                "num_frames": kw["num_frames"],
                "num_history_frames": kw["num_history_frames"],
                "seed": kw["seed"],
                "height": kw["height"],
                "width": kw["width"],
                "steps": kw["num_inference_steps"],
                "cfg_scale": kw["cfg_scale"],
                "use_history_condition_noise_in_inference": kw["use_history_condition_noise_in_inference"],
                "seconds": round(time.perf_counter() - t0, 3),
            }
        )
        ctx["window"] += 1
        return out

    cls.__call__ = call

    real_save = ref.save_video

    def save_video(video, output_path, fps, quality):
        v = video.detach().to(torch.float32).cpu()
        frames = [_normalize_to_uint8(v[0, :, i].permute(1, 2, 0).numpy()) for i in range(v.shape[2])]
        np.save(ctx["out"] / "frames_u8.npy", np.stack(frames))
        return real_save(video, output_path=str(ctx["out"] / "rollout.mp4"), fps=fps, quality=quality)

    ref.save_video = save_video

    for spec_tag in args.tags:
        tag, *modes = spec_tag.split(":")
        out = out_root / tag
        out.mkdir(parents=True, exist_ok=True)
        ctx.update(
            window=0,
            out=out,
            log=[],
            noise_files=[],
            steps_in_window=0,
            noise_seed=1042 if "noise1042" in modes else 42,
            histnoise=False if "nohistnoise" in modes else None,
            nocarry="nocarry" in modes,
            quantcarry="quantcarry" in modes,
        )
        sample = ref.prepare_sample_for_rollout(dataset[0], 0, pipe, rargs)
        np.savez(
            out / "inputs.npz",
            first_frame=sample["video"][:, :, 0].float().cpu().numpy(),
            action=sample["action"].float().cpu().numpy(),
        )
        print(f"=== REF variant {spec_tag}", flush=True)
        t0 = time.perf_counter()
        ref._run_autoregressive(pipe=pipe, sample=sample, args=rargs)
        meta = {
            "variant": spec_tag,
            "dtype": args.dtype,
            "noise_files": ctx["noise_files"],
            "windows": ctx["log"],
            "total_frames": sample["total_frames"],
            "rollout_seconds": round(time.perf_counter() - t0, 2),
            "resolved_args": resolved,
        }
        (out / "run.json").write_text(json.dumps(meta, indent=2))
        print(json.dumps({k: v for k, v in meta.items() if k != "resolved_args"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
