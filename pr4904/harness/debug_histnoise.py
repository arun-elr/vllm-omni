"""Trace the reference's history re-noising for one window (diagnostic)."""

import importlib.util
import os
import sys
from pathlib import Path

import torch

S = Path(os.environ.get("S", "."))
REF = S / "bwm-ref"
os.chdir(REF)
sys.path.insert(0, str(REF))
spec = importlib.util.spec_from_file_location("ref_infer", REF / "scripts" / "infer.py")
ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)
import wan_video_action.pipelines.wan_video_action as wva

sys.argv = [
    "infer.py", "--config", "configs/infer/infer.yaml",
    "--model_paths", str(S / "weights/Wan2.2-TI2V-5B"),
    "--ckpt_path", str(S / "weights/BLM/step-12000.safetensors"),
    "--dataset_base_path", "demo", "--dataset_metadata_path", "demo/demo.jsonl",
    "--action_stat_path", "demo/stat.json", "--output_path", "/tmp/dbg",
    "--height", "480", "--width", "640", "--max_samples", "1",
]
args = ref.parse_args()
pipe = ref.build_pipeline(args)
real = wva._prepare_history_condition_latents


def traced(self, inputs_shared, *, use_history_condition_noise_in_inference):
    ffl = inputs_shared.get("first_frame_latents")
    before = inputs_shared["latents"].clone()
    print("flag", use_history_condition_noise_in_inference, "mode", getattr(self, "action_injection_mode", None))
    print("first_frame_latents", None if ffl is None else tuple(ffl.shape), "latents", tuple(before.shape))
    print("scheduler timesteps", len(self.scheduler.timesteps), self.scheduler.timesteps[:3].tolist())
    noise = inputs_shared["noise"]
    print("latents shares storage with noise:", inputs_shared["latents"].data_ptr() == noise.data_ptr(),
          "| noise[:, :, :3] equals history latents:", torch.equal(noise[:, :, :3], ffl))
    out = real(self, inputs_shared, use_history_condition_noise_in_inference=use_history_condition_noise_in_inference)
    cond, history_t = out
    after = inputs_shared["latents"]
    print("history_t", history_t, "latents changed per frame:", [(before[:, :, i] != after[:, :, i]).any().item() for i in range(4)])
    if cond is not None:
        noise = inputs_shared["noise"]
        print("cond frames 1.. equal noise:", [torch.equal(cond[:, :, i], noise[:, :, i]) for i in range(1, history_t)])
    raise SystemExit(0)


wva._prepare_history_condition_latents = traced
sample = ref.prepare_sample_for_rollout(ref.build_infer_dataset(args)[0], 0, pipe, args)
ref._run_autoregressive(pipe=pipe, sample=sample, args=args)
