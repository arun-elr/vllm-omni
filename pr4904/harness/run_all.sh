#!/bin/bash
# PR #4904 BWM parity run: item-2 reruns, then the controlled PR/reference matrix.
# Usage (on the GPU host, from the scratch dir): bash harness/run_all.sh [stage...]
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
S=$(pwd)
H=$S/harness
PR=$S/vllm-omni-a548404
FIX=$S/vllm-omni-histnoise
REF=$S/bwm-ref
OMNI=$S/venv-omni/bin/python
REFPY=$S/venv-ref/bin/python
W=$S/weights
RUNS=$S/runs
LOCK=${LOCK:-/tmp/h200.lock}
DIFFUSERS_BASE=$HOME/.cache/huggingface/hub/models--Wan-AI--Wan2.2-TI2V-5B-Diffusers/snapshots/b8fff7315c768468a5333511427288870b2e9635
mkdir -p $S/logs $S/env $RUNS $S/item2
stages=${*:-env unit assemble item2 noise pr ref quant compare}

for st in $stages; do
case $st in
env)
  { nvidia-smi; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv; } > $S/env/nvidia-smi.txt
  $S/venv-omni/bin/python -m pip --version >/dev/null 2>&1 || true
  VIRTUAL_ENV=$S/venv-omni uv pip freeze > $S/env/pip-freeze-vllm-omni.txt
  VIRTUAL_ENV=$S/venv-ref uv pip freeze > $S/env/pip-freeze-reference.txt
  for py in $OMNI $REFPY; do
    $py -c "import sys, torch; print(sys.executable.split('scratch-pr4904/')[-1], 'python', sys.version.split()[0], 'torch', torch.__version__, 'cuda', torch.version.cuda, 'cudnn', torch.backends.cudnn.version(), 'gpu', torch.cuda.get_device_name(0))"
  done > $S/env/torch.txt
  $OMNI -c "import vllm, vllm_omni, diffusers; print('vllm', vllm.__version__, 'vllm_omni', vllm_omni.__version__, 'diffusers', diffusers.__version__)" >> $S/env/torch.txt 2>/dev/null
  $REFPY -c "import importlib.metadata as m; print('diffsynth', m.version('diffsynth'))" >> $S/env/torch.txt
  cat $S/env/revisions.txt $S/env/torch.txt
  ;;
unit)
  (cd $PR && $OMNI -m pytest tests/diffusion/models/bwm/ -v -p no:cacheprovider) 2>&1 | tee $S/logs/unit-tests.log
  ;;
assemble)
  # The PR's own assembly script, against the pinned diffusers base.
  (cd $PR && $OMNI examples/offline_inference/bwm/download_bwm.py --output-dir $S/models/BWM --base-dir $DIFFUSERS_BASE) 2>&1 | tee $S/logs/assemble.log
  ;;
item2)
  # The PR's documented command, unmodified: full 143-frame episode, native 480x640, eager.
  (cd $PR && /usr/bin/time -v flock $LOCK $OMNI examples/offline_inference/bwm/bwm_world_model.py \
      --model $S/models/BWM --episode-dir $REF/demo --episode 0 --output $S/item2/bwm_rollout_143.mp4 --enforce-eager) \
      2>&1 | tee $S/logs/item2-rollout-143.log
  ;;
noise)
  flock $LOCK $OMNI $H/make_noise.py --out $S/noise 2>&1 | tee $S/logs/noise.log
  # Episode 0 trimmed to 141 frames: the reference floors 143 to the 4n+1 grid
  # (align_num_frames), the PR example does not. Same frames and actions,
  # only the metadata end_frame changes, so both sides roll out 57/57/53.
  mkdir -p $S/episode141
  ln -sfn $REF/demo/adjust_bottle $S/episode141/adjust_bottle
  cp $REF/demo/stat.json $S/episode141/stat.json
  head -1 $REF/demo/demo.jsonl | sed 's/"length": 143/"length": 141/; s/"end_frame": 142/"end_frame": 140/' > $S/episode141/demo.jsonl
  cat $S/episode141/demo.jsonl
  ;;
pr)
  common="--example $PR/examples/offline_inference/bwm/bwm_world_model.py --model $S/models/BWM --episode-dir $S/episode141 --out-dir $RUNS"
  export BWM_DUMP_DIR=$RUNS
  PYTHONPATH=$H/hook flock $LOCK $OMNI $H/pr_rollout.py $common \
      --tags pr-bf16 pr-bf16-repeat pr-bf16-nocarry:nocarry 2>&1 | tee $S/logs/pr-bf16.log
  PYTHONPATH=$H/hook flock $LOCK $OMNI $H/pr_rollout.py $common --dtype float32 \
      --tags pr-fp32 2>&1 | tee $S/logs/pr-fp32.log
  PYTHONPATH=$H/hook:$FIX flock $LOCK $OMNI $H/pr_rollout.py $common \
      --tags prfix-bf16 prfix-bf16-nocarry:nocarry 2>&1 | tee $S/logs/prfix-bf16.log
  PYTHONPATH=$H/hook:$FIX flock $LOCK $OMNI $H/pr_rollout.py $common --dtype float32 \
      --tags prfix-fp32 2>&1 | tee $S/logs/prfix-fp32.log
  ;;
ref)
  common="--ref-repo $REF --model-paths $W/Wan2.2-TI2V-5B --ckpt $W/BLM/step-12000.safetensors --noise-dir $S/noise --out-dir $RUNS"
  flock $LOCK $REFPY $H/ref_rollout.py $common --dtype bf16 \
      --tags ref-bf16 ref-bf16-repeat ref-bf16-nohistnoise:nohistnoise ref-bf16-noise1042:noise1042 ref-bf16-nocarry:nocarry \
      2>&1 | tee $S/logs/ref-bf16.log
  flock $LOCK $REFPY $H/ref_rollout.py $common --dtype fp32 \
      --tags ref-fp32 ref-fp32-nohistnoise:nohistnoise 2>&1 | tee $S/logs/ref-fp32.log
  ;;
quant)
  # Carry-over handoff check: reference with the PR's uint8 history representation.
  common="--ref-repo $REF --model-paths $W/Wan2.2-TI2V-5B --ckpt $W/BLM/step-12000.safetensors --noise-dir $S/noise --out-dir $RUNS"
  flock $LOCK $REFPY $H/ref_rollout.py $common --dtype bf16 --tags ref-bf16-quantcarry:quantcarry 2>&1 | tee $S/logs/ref-bf16-quant.log
  flock $LOCK $REFPY $H/ref_rollout.py $common --dtype fp32 --tags ref-fp32-quantcarry:quantcarry 2>&1 | tee $S/logs/ref-fp32-quant.log
  ;;
compare)
  $OMNI $H/compare.py --runs $RUNS --dump $RUNS --noise-dir $S/noise --out $S/results.json 2>&1 | tee $S/logs/compare.log
  ;;
esac
done
