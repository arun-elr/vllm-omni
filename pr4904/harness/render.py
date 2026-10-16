"""Render small artifacts from results.json and the saved uint8 frames."""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

COLS = [
    ("ref-bf16", "reference"),
    ("pr-bf16", "vllm-omni a54840407"),
    ("diff", "|PR - reference| x4"),
    ("ref-bf16-noise1042", "reference, other noise (neg. control)"),
]
CHART_PAIRS = [
    ("pr-bf16", "ref-bf16", "bf16: PR vs reference"),
    ("pr-bf16", "ref-bf16-quantcarry", "bf16: PR vs reference, uint8 carry-over matched"),
    ("pr-fp32", "ref-fp32", "fp32: PR vs reference"),
    ("pr-fp32", "ref-fp32-quantcarry", "fp32: PR vs reference, uint8 carry-over matched"),
    ("ref-bf16", "ref-fp32", "yardstick: reference bf16 vs fp32"),
    ("pr-bf16", "pr-fp32", "yardstick: PR bf16 vs fp32"),
    ("ref-bf16", "ref-bf16-noise1042", "negative: reference, other noise"),
    ("pr-bf16", "pr-bf16-nocarry", "negative: PR, carry-over broken"),
]


def small(frame, w=320):
    im = Image.fromarray(frame)
    return np.asarray(im.resize((w, int(im.height * w / im.width)), Image.BILINEAR))


def pair(results, a, b):
    for p in results["pairs"]:
        if p["a"] == a and p["b"] == b:
            return p
    return None


def chart(results, out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    n = len(CHART_PAIRS)
    width = 0.8 / n
    for i, (a, b, label) in enumerate(CHART_PAIRS):
        p = pair(results, a, b)
        if p is None:
            continue
        ws = p["windows"]
        x = np.arange(len(ws)) + (i - n / 2 + 0.5) * width
        ps = [70.0 if w["psnr_db"] is None else w["psnr_db"] for w in ws]
        rel = [max(w["latent"]["rel_mean_abs"], 1e-5) for w in ws]
        hatch = ["//" if w["pixels_identical"] else None for w in ws]
        b0 = axes[0].bar(x, ps, width, label=label)
        b1 = axes[1].bar(x, rel, width, label=label)
        for bars in (b0, b1):
            for bar, h in zip(bars, hatch):
                if h:
                    bar.set_hatch(h)
                    bar.set_alpha(0.35)
    for ax in axes:
        ax.set_xticks(range(3))
        ax.set_xticklabels(["window 0\n(frames 1-48)", "window 1\n(49-96)", "window 2\n(97-140)"])
    axes[0].set_ylabel("PSNR of generated frames (dB, higher = closer)")
    axes[0].set_title("Decoded frames, per window")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("mean |latent diff| / mean |latent| (lower = closer)")
    axes[1].set_title("Final denoised latents, per window")
    axes[1].legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.text(0.99, 0.01, "hatched = bit-identical (drawn at the axis cap)", ha="right", fontsize=7.5)
    fig.suptitle("BWM rollout, vllm-omni a54840407 vs reference: same checkpoint, noise, actions, 480x640, Euler 50 steps, 1x H200")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def label_row(images, labels, title=None):
    fig, axes = plt.subplots(1, len(images), figsize=(3.2 * len(images), 2.75))
    for ax, im, lab in zip(np.atleast_1d(axes), images, labels):
        ax.imshow(im)
        ax.set_title(lab, fontsize=9)
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig


def grids(runs, out_dir):
    frames = {t: np.load(runs / t / "frames_u8.npy") for t, _ in COLS if t != "diff"}
    d = np.abs(frames["pr-bf16"].astype(np.int16) - frames["ref-bf16"].astype(np.int16)) * 4
    frames["diff"] = np.clip(d, 0, 255).astype(np.uint8)
    n = frames["ref-bf16"].shape[0]
    picks = [(1, "window 0, first generated frame (1)"), (48, "window 0, last frame (48)"),
             (49, "window 1, first frame (49)"), (96, "window 1, last frame (96)"),
             (97, "window 2, first frame (97)"), (n - 1, f"window 2, last frame ({n - 1})")]
    rows = []
    for idx, title in picks:
        fig = label_row([small(frames[t][idx]) for t, _ in COLS], [lab for _, lab in COLS], title)
        fig.canvas.draw()
        rows.append(np.asarray(fig.canvas.buffer_rgba())[..., :3])
        plt.close(fig)
    imageio.imwrite(out_dir / "pr4904_window_grid.png", np.concatenate(rows, axis=0))
    fig = label_row(
        [small(frames["ref-bf16"][0])] + [small(frames[t][1]) for t, _ in COLS],
        ["input frame 0 (shared)"] + [lab for _, lab in COLS],
        "First generated frame",
    )
    fig.savefig(out_dir / "pr4904_first_frame.png", dpi=100)
    plt.close(fig)
    with imageio.get_writer(out_dir / "pr4904_side_by_side.mp4", fps=24, codec="libx264", quality=6) as wr:
        for i in range(n):
            wr.append_data(np.concatenate([small(frames[t][i], 320) for t, _ in COLS], axis=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    results = json.loads(args.results.read_text())
    chart(results, args.out / "pr4904_parity_chart.png")
    grids(args.runs, args.out)


if __name__ == "__main__":
    main()
