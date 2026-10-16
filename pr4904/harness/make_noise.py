"""Generate the shared initial noise once and save it.

Draws exactly what the PR pipeline draws for each window (the diffusion
worker seeds ``torch.Generator(device="cuda")`` with ``seed + window`` and
``prepare_latents`` calls ``randn_tensor(shape, generator, device, float32)``),
so the PR runs unmodified and the reference gets the same tensor injected.
The PR-side hook dumps the noise the pipeline actually drew; compare.py
asserts it is bit-identical to these files.
"""

import argparse
import json
from pathlib import Path

import torch
from diffusers.utils.torch_utils import randn_tensor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 1042])
    p.add_argument("--latent-frames", type=int, nargs="+", default=[15, 15, 14])
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=640)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for seed in args.seeds:
        for w, t in enumerate(args.latent_frames):
            shape = (1, 48, t, args.height // 16, args.width // 16)
            g = torch.Generator(device="cuda").manual_seed(seed + w)
            noise = randn_tensor(shape, generator=g, device=torch.device("cuda"), dtype=torch.float32).cpu()
            name = f"seed{seed}_w{w}.pt"
            torch.save(noise, args.out / name)
            manifest[name] = {"seed": seed + w, "shape": list(shape), "mean": noise.mean().item(), "std": noise.std().item()}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
