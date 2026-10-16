"""Compare BWM rollouts: noise/input checks, per-window latents and pixels.

Run dirs (``<runs>/<tag>/``) hold ``frames_u8.npy`` (the uint8 frames each
implementation writes to its mp4), ``latents_w{k}.pt`` (final denoised
latents), ``noise_pred0_w{k}.pt`` (first DiT output) and ``inputs.npz``.
Latents are compared on the generated latent frames only (index >= 3; the
first 3 latent frames are the 9 pinned history frames).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from skimage.metrics import structural_similarity

HISTORY_LATENTS = 3
HISTORY_FRAMES = 9
FUTURE = 48

PAIRS = [
    ("primary", "PR vs reference (as released)", "pr-bf16", "ref-bf16"),
    ("primary", "PR vs reference, fp32", "pr-fp32", "ref-fp32"),
    ("carry", "PR vs reference with PR-style uint8 history carry-over, bf16", "pr-bf16", "ref-bf16-quantcarry"),
    ("carry", "PR vs reference with PR-style uint8 history carry-over, fp32", "pr-fp32", "ref-fp32-quantcarry"),
    ("yardstick", "reference bf16 vs reference fp32 (dtype change only)", "ref-bf16", "ref-fp32"),
    ("yardstick", "PR bf16 vs PR fp32 (dtype change only)", "pr-bf16", "pr-fp32"),
    ("determinism", "PR bf16, rerun", "pr-bf16", "pr-bf16-repeat"),
    ("determinism", "reference bf16, rerun", "ref-bf16", "ref-bf16-repeat"),
    ("negative", "reference, different initial noise", "ref-bf16", "ref-bf16-noise1042"),
    ("negative", "PR, history carry-over broken", "pr-bf16", "pr-bf16-nocarry"),
    ("negative", "reference, history carry-over broken", "ref-bf16", "ref-bf16-nocarry"),
    ("reference-flag", "reference as released vs use_history_condition_noise_in_inference=False, bf16", "ref-bf16", "ref-bf16-nohistnoise"),
    ("reference-flag", "reference as released vs use_history_condition_noise_in_inference=False, fp32", "ref-fp32", "ref-fp32-nohistnoise"),
    ("counterfactual", "PR + history re-noising (what the flag intends) vs reference, bf16", "prfix-bf16", "ref-bf16"),
    ("counterfactual", "PR + history re-noising (what the flag intends) vs reference, fp32", "prfix-fp32", "ref-fp32"),
    ("counterfactual", "PR + history re-noising, history carry-over broken", "prfix-bf16", "prfix-bf16-nocarry"),
]


def window_ranges(n_frames):
    out, start = [], 1
    while start < n_frames:
        end = min(start + FUTURE, n_frames)
        out.append((start, end))
        start = end
    return out


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return None if mse == 0 else float(10 * np.log10(255.0**2 / mse))


def ssim_mean(a, b):
    return float(np.mean([structural_similarity(x, y, channel_axis=2, data_range=255) for x, y in zip(a, b)]))


def tensor_diff(a, b):
    a, b = a.float(), b.float()
    d = (a - b).abs()
    return {
        "max_abs": d.max().item(),
        "mean_abs": d.mean().item(),
        "rel_mean_abs": (d.mean() / b.abs().mean()).item(),
        "bit_exact": bool(torch.equal(a, b)),
    }


def compare_pair(runs, a, b):
    fa, fb = np.load(runs / a / "frames_u8.npy"), np.load(runs / b / "frames_u8.npy")
    assert fa.shape == fb.shape, (a, fa.shape, b, fb.shape)
    res = {"frames": int(fa.shape[0]), "frame0_identical": bool(np.array_equal(fa[0], fb[0])), "windows": []}
    for w, (s, e) in enumerate(window_ranges(fa.shape[0])):
        lat_a, lat_b = torch.load(runs / a / f"latents_w{w}.pt"), torch.load(runs / b / f"latents_w{w}.pt")
        np_a, np_b = torch.load(runs / a / f"noise_pred0_w{w}.pt"), torch.load(runs / b / f"noise_pred0_w{w}.pt")
        res["windows"].append(
            {
                "window": w,
                "frames": [s, e],
                "latent": tensor_diff(lat_a[:, :, HISTORY_LATENTS:], lat_b[:, :, HISTORY_LATENTS:]),
                "first_dit_output": tensor_diff(np_a[:, :, HISTORY_LATENTS:], np_b[:, :, HISTORY_LATENTS:]),
                "psnr_db": psnr(fa[s:e], fb[s:e]),
                "psnr_first_frame_db": psnr(fa[s], fb[s]),
                "ssim": ssim_mean(fa[s:e], fb[s:e]),
                "pixels_identical": bool(np.array_equal(fa[s:e], fb[s:e])),
            }
        )
    return res


def noise_checks(runs, dump, noise_dir):
    out = {}
    for tag_dir in sorted(p for p in dump.iterdir() if p.is_dir()):
        files = sorted(tag_dir.glob("noise_w*.pt"))
        if not files:
            continue
        checks = []
        for f in files:
            w = int(f.stem.split("_w")[1])
            ref = torch.load(noise_dir / f"seed42_w{w}.pt")
            got = torch.load(f)
            checks.append({"window": w, "shape": list(got.shape), "bit_identical": bool(torch.equal(got, ref))})
        out[tag_dir.name] = checks
    return out


def input_checks(runs, pr_tag, ref_tag):
    p, r = np.load(runs / pr_tag / "inputs.npz"), np.load(runs / ref_tag / "inputs.npz")
    pr_frame = p["first_frame"]  # (H, W, 3) uint8
    ref_frame = r["first_frame"][0]  # (3, H, W) float in [-1, 1]
    pr_float = torch.from_numpy(pr_frame).permute(2, 0, 1).float() / 127.5 - 1.0
    ref_float = torch.from_numpy(ref_frame).float()
    pr_act = torch.from_numpy(p["action"]).float()
    ref_act = torch.from_numpy(r["action"][0]).float()
    n = min(pr_act.shape[0], ref_act.shape[0])
    return {
        "first_frame_max_abs_diff": (pr_float - ref_float).abs().max().item(),
        "first_frame_bf16_identical": bool(torch.equal(pr_float.bfloat16(), ref_float.bfloat16())),
        "action_frames_pr_ref": [int(pr_act.shape[0]), int(ref_act.shape[0])],
        "action_bf16_identical": bool(torch.equal(pr_act[:n].bfloat16().float(), ref_act[:n])),
        "action_max_abs_diff": (pr_act[:n] - ref_act[:n]).abs().max().item(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--dump", type=Path, required=True)
    ap.add_argument("--noise-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--noise-checks-from", type=Path, default=None,
                    help="results JSON from the GPU host, when the PR-side noise dumps were not copied")
    args = ap.parse_args()

    checks = noise_checks(args.runs, args.dump, args.noise_dir)
    if not checks and args.noise_checks_from:
        checks = json.loads(args.noise_checks_from.read_text())["noise_checks_pr_side"]
    results = {
        "noise_checks_pr_side": checks,
        "input_checks": input_checks(args.runs, "pr-bf16", "ref-bf16"),
        "pairs": [],
    }
    for kind, label, a, b in PAIRS:
        if not ((args.runs / a / "frames_u8.npy").exists() and (args.runs / b / "frames_u8.npy").exists()):
            print(f"skip {a} vs {b}: missing run")
            continue
        r = compare_pair(args.runs, a, b)
        r.update(kind=kind, label=label, a=a, b=b)
        results["pairs"].append(r)
        ws = "  ".join(
            f"w{x['window']}: lat mean {x['latent']['mean_abs']:.4g} max {x['latent']['max_abs']:.3g}"
            f" | PSNR {'identical' if x['psnr_db'] is None else format(x['psnr_db'], '.2f')} SSIM {x['ssim']:.4f}"
            for x in r["windows"]
        )
        print(f"[{kind}] {label} ({a} vs {b})\n    {ws}")
    args.out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
