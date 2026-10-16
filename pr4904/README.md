# BWM controlled parity: vllm-omni PR #4904 vs the reference implementation

Evidence for https://github.com/vllm-project/vllm-omni/pull/4904 (head `a54840407e0c07deef2bd22f4df11763bcb83ffd`).

## Setup

| | vllm-omni (PR) | Reference |
|---|---|---|
| Revision | `a54840407e0c07deef2bd22f4df11763bcb83ffd`, unmodified | boundless-large-model/boundless-world-model `cc0bc74ec24867975f1dd26939f3864343368367` (code identical to the July 2 head `44acfd1b`; later commits are README and figures only) |
| Entry point | `examples/offline_inference/bwm/bwm_world_model.py` `main()`, called as is | `scripts/infer.py`: its own arg parsing, dataset loader, `build_pipeline` and `_run_autoregressive` |
| Checkpoint | `BLM-Lab/Boundless-World-Model@738a8d3` `step-12000.safetensors`, assembled with the PR's `download_bwm.py` onto `Wan-AI/Wan2.2-TI2V-5B-Diffusers@b8fff73` | same checkpoint onto `Wan-AI/Wan2.2-TI2V-5B@921dbaf` (Wan-native DiT + `Wan2.2_VAE.pth`) |
| Episode / actions | `demo/` episode 0 (adjust_bottle), 141 frames, p01/p99 normalized eef_abs actions | same (the reference floors 143 to the 4n+1 grid, 141; the PR side gets the same 141 frames through its metadata) |
| Windows | 57, 57, 53 frames; 9 history frames (frame 0 + 8 most recent); seeds 42, 43, 44 | same |
| Initial noise | the pipeline's own draw, `torch.Generator("cuda").manual_seed(42 + window)` | the same tensor, injected from file in place of `generate_noise` (pre-generated once by `make_noise.py`; bit-equality with the PR's draw asserted in `results.json`) |
| Resolution | 480x640 | 480x640 (`--height 480 --width 640`) |
| Scheduler | `WanEulerScheduler`, shift 5, 50 steps | DiffSynth `FlowMatchScheduler("Wan")`, shift 5, 50 steps (same sigma schedule) |
| dtype | bf16 (default); fp32 via `Omni(dtype="float32")` | bf16 (`build_pipeline`); fp32 rebuild with `torch.float32` |
| Attention | FlashAttention (bf16), SDPA (fp32, automatic) | SDPA |
| Software | torch 2.11.0+cu130, vllm 0.24.0, diffusers 0.38.0 | torch 2.8.0+cu128, diffsynth 2.0.11 (reference README install) |
| GPU | 1x H200 | same H200 |

## Reproduce

On a 1-GPU host, with this directory's `harness/` next to checkouts of both repos (`vllm-omni-a548404/`, `bwm-ref/`) in a scratch dir:

```bash
bash harness/setup_h200.sh       # two throwaway venvs
bash harness/download_h200.sh    # pinned weights
bash harness/run_all.sh env unit assemble item2 noise pr ref quant compare
python harness/render.py --runs runs --results results.json --out .
```

`run_all.sh` holds every exact command; `logs/` has their output, `env/` the pip freezes, nvidia-smi and revisions.

## Variants (`runs/<tag>/run.json`)

| Tag | What |
|---|---|
| `pr-bf16`, `pr-fp32` | PR, unmodified |
| `ref-bf16`, `ref-fp32` | reference, unmodified except injected noise |
| `*-repeat` | rerun, determinism check |
| `ref-*-nohistnoise` | reference with `use_history_condition_noise_in_inference=False` (`infer.py` hardcodes True) |
| `ref-*-quantcarry` | reference carrying generated history frames as truncated uint8, like the PR example |
| `ref-bf16-noise1042` | negative control: other initial noise |
| `*-nocarry` | negative control: history of windows 1 and 2 replaced by copies of frame 0 |
| `prfix-*` | counterfactual: PR plus history re-noising (`harness/prfix_histnoise.patch`), i.e. what the reference flag intends; not proposed |

## Notes

* The reference's `use_history_condition_noise_in_inference=True` is a no-op in the released rollout: the initial `latents` tensor is the `noise` tensor (same storage), the history VAE latents are written into it in place, so the "re-noised" history equals the clean history (`logs/debug-histnoise.log`). The reference with the flag off is bit-identical to the reference as released (`results.json`, kind `reference-flag`).
* The PR example carries generated history frames as truncated uint8; the reference carries floats. Matching that one representation (`ref-*-quantcarry`) brings fp32 windows 1 and 2 to the window-0 level.
* On the full 143-frame episode the PR example also generates frames 141 and 142 (last window padded to 57 with the final action); the reference stops at 141.

## Recipe command: output and peak memory

`harness/mem_run.sh` runs the recipe's command as documented (480x640 demo input, default compile), the same command with `--enforce-eager`, and the same command on a 672x896 copy of the demo episode (video upscaled bilinear, same actions). Peaks come from `harness/memhook/sitecustomize.py`, a read-only hook in the diffusion worker that logs `torch.cuda.max_memory_allocated` and `max_memory_reserved` after each window. The runner resets peak stats before each forward, so these are per-forward peaks (model load excluded); `harness/summarize_mem.py` reports the max over windows and the per-window values. nvidia-smi `memory.used` is sampled every 200 ms over the full run (whole device, includes load). Results are in `mem/summary-*.json`, with full logs alongside.

## Review 3 checks (output_type, short tails)

`harness/gpu_check.sh` reruns the documented 143-frame command on the fixed code and compares it to the `mem/` runs on the previous code; `harness/latent_e2e.py` sends one window with `output_type="np"` and `"latent"` (same seed), decodes the returned latents with the model's VAE and compares to the np output; `harness/compile_determinism.sh` reruns the previous code with the default compile to measure run-to-run spread. Results: `check3/summary.json`, `check3/latent_e2e.json`, `check3/compile_determinism.json`, `check3/tail52.json` (a 52-frame episode, eager), and the unit-test logs `check3/tests-*.log`.
