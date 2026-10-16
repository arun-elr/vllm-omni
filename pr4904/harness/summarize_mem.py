"""Summarize one mem_run.sh case: output frames/dims, per-forward allocator
peaks (max over windows) and the sampled nvidia-smi whole-device peak.

Usage: python summarize_mem.py <mem_dir> <case>
"""
import csv
import json
import sys

import imageio.v2 as imageio

out, tag = sys.argv[1], sys.argv[2]
frames = [f for f in imageio.get_reader(f"{out}/bwm_rollout_{tag}.mp4")]
recs = [json.loads(l) for l in open(f"{out}/{tag}.jsonl")]
smi = [int(row[1]) for row in csv.reader(open(f"{out}/smi-{tag}.csv")) if len(row) > 1 and row[1].strip().isdigit()]
s = {
    "case": tag,
    "output_frames": len(frames),
    "output_hw": list(frames[0].shape[:2]),
    "windows": len(recs),
    "request_hw": [recs[0]["height"], recs[0]["width"]],
    "dtype": recs[0]["dtype"],
    # The runner resets peak stats before each forward, so each record is a
    # per-forward peak (model load excluded); report the max over windows.
    "torch_per_forward_peak_allocated_gib_max_over_windows": max(r["max_memory_allocated_gib"] for r in recs),
    "torch_per_forward_peak_reserved_gib_max_over_windows": max(r["max_memory_reserved_gib"] for r in recs),
    "torch_per_forward_peak_allocated_gib_per_window": [r["max_memory_allocated_gib"] for r in recs],
    "torch_per_forward_peak_reserved_gib_per_window": [r["max_memory_reserved_gib"] for r in recs],
    # Whole-device memory.used sampled every 200 ms over the full run (includes load).
    "nvidia_smi_peak_used_gib": round(max(smi) / 1024, 2),
    "nvidia_smi_samples": len(smi),
}
json.dump(s, open(f"{out}/summary-{tag}.json", "w"), indent=2)
print(json.dumps(s))
