"""End-to-end check of output_type="latent" through Omni for BWM (PR #4904).

Runs the first rollout window of demo episode 0 twice with the same seed:
once with output_type="np" and once with output_type="latent". Checks that
the latent output arrives as a latent tensor, then decodes it with the
model's VAE (same de-normalization as the pipeline) and compares the result
to the np output.

Usage: python latent_e2e.py <model_dir> <episode_dir> <example_py> <out_json>
"""

import importlib.util
import json
import sys

import numpy as np
import torch


def main():
    model_dir, episode_dir, example_py, out_json = sys.argv[1:5]
    spec = importlib.util.spec_from_file_location("bwm_world_model", example_py)
    ex = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ex)

    from pathlib import Path

    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    first_frame, action, meta = ex.load_episode(Path(episode_dir), 0)
    height, width = first_frame.shape[:2]
    hist_idx = ex.history_indices(1, ex.NUM_HISTORY_FRAMES)
    cond = np.concatenate([action[hist_idx], action[1 : 1 + ex.NUM_FRAMES - ex.NUM_HISTORY_FRAMES]], axis=0)
    history = np.stack([first_frame] * ex.NUM_HISTORY_FRAMES, axis=0)

    omni = Omni(model=model_dir, model_class_name="BoundlessWorldModelPipeline", enforce_eager=True)

    def run(output_type):
        outputs = omni.generate(
            {"prompt": "", "multi_modal_data": {"video": history, "action": cond}},
            OmniDiffusionSamplingParams(
                height=height,
                width=width,
                num_frames=cond.shape[0],
                num_inference_steps=50,
                guidance_scale=1.0,
                seed=42,
                output_type=output_type,
            ),
        )
        out = outputs[0]
        if getattr(out, "is_pipeline_output", False) and getattr(out, "request_output", None) is not None:
            out = out.request_output
        return out.images[0]

    video_np = np.asarray(run("np"))
    latents = run("latent")

    rec = {
        "np_type": type(video_np).__name__,
        "np_shape": list(video_np.shape),
        "latent_type": type(latents).__name__,
        "latent_shape": list(latents.shape) if isinstance(latents, torch.Tensor) else None,
        "latent_dtype": str(latents.dtype) if isinstance(latents, torch.Tensor) else None,
        "latent_finite": bool(torch.isfinite(latents).all()) if isinstance(latents, torch.Tensor) else None,
    }

    from diffusers import AutoencoderKLWan
    from diffusers.video_processor import VideoProcessor

    # Plain diffusers VAE (same weights); the distributed wrapper needs the worker's process group.
    vae = AutoencoderKLWan.from_pretrained(model_dir, subfolder="vae", torch_dtype=torch.bfloat16).to("cuda")
    z = latents.to("cuda", vae.dtype)
    mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(z.device, z.dtype)
    std_inv = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(z.device, z.dtype)
    with torch.no_grad():
        decoded = vae.decode(z / std_inv + mean, return_dict=False)[0]
    decoded_np = VideoProcessor(vae_scale_factor=16).postprocess_video(decoded.float().cpu(), output_type="np")

    diff = np.abs(decoded_np - video_np)
    mse = float((diff**2).mean())
    rec.update(
        {
            "decoded_shape": list(decoded_np.shape),
            "decoded_vs_np_max_abs": float(diff.max()),
            "decoded_vs_np_mean_abs": float(diff.mean()),
            "decoded_vs_np_psnr_db": float("inf") if mse == 0 else float(10 * np.log10(1.0 / mse)),
        }
    )
    print(json.dumps(rec))
    with open(out_json, "w") as f:
        json.dump(rec, f, indent=2)


if __name__ == "__main__":
    main()
