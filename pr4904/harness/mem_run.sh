#!/bin/bash
# Peak-memory and output check for the recipe command (PR #4904), from the scratch dir.
# Cases: 480x640 demo input as documented (default, compiled), same with --enforce-eager,
# and a 672x896 copy of the demo episode (the example takes its resolution from the input frame).
set -euo pipefail
S=$(pwd)
PR=$S/vllm-omni-a548404
OMNI=$S/venv-omni/bin/python
LOCK=${LOCK:-/tmp/h200.lock}
OUT=$S/mem
mkdir -p $OUT
exec 9>$LOCK
flock 9

# 672x896 episode: same frames upscaled (bilinear), same actions and stats.
FFMPEG=$($OMNI -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())")
D=$S/demo672
V=adjust_bottle/videos/chunk-000/episode_000040.mp4
if [ ! -f $D/$V ]; then
  mkdir -p $D/adjust_bottle/videos/chunk-000
  ln -sfn $S/bwm-ref/demo/adjust_bottle/data $D/adjust_bottle/data
  cp $S/bwm-ref/demo/demo.jsonl $S/bwm-ref/demo/stat.json $D/
  $FFMPEG -loglevel error -y -i $S/bwm-ref/demo/$V -vf scale=896:672:flags=bilinear -c:v libx264 -crf 10 -pix_fmt yuv420p $D/$V
fi

run() {
  local tag=$1 epdir=$2; shift 2
  rm -f $OUT/$tag.jsonl
  nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader,nounits -lms 200 > $OUT/smi-$tag.csv &
  local smi=$!
  (cd $PR && BWM_MEM_LOG=$OUT/$tag.jsonl PYTHONPATH=$S/harness/memhook /usr/bin/time -v \
      $OMNI examples/offline_inference/bwm/bwm_world_model.py \
      --model $S/models/BWM --episode-dir $epdir --episode 0 --output $OUT/bwm_rollout_$tag.mp4 "$@") \
      > $OUT/log-$tag.log 2>&1
  kill $smi
  $OMNI - $OUT $tag <<'EOF'
import csv, json, sys, imageio.v2 as imageio
out, tag = sys.argv[1], sys.argv[2]
r = imageio.get_reader(f"{out}/bwm_rollout_{tag}.mp4")
frames = [f for f in r]
recs = [json.loads(l) for l in open(f"{out}/{tag}.jsonl")]
smi = [int(row[1]) for row in csv.reader(open(f"{out}/smi-{tag}.csv")) if len(row) > 1 and row[1].strip().isdigit()]
s = {
    "case": tag,
    "output_frames": len(frames),
    "output_hw": list(frames[0].shape[:2]),
    "windows": len(recs),
    "request_hw": [recs[0]["height"], recs[0]["width"]],
    "dtype": recs[0]["dtype"],
    "torch_max_memory_allocated_gib": recs[-1]["max_memory_allocated_gib"],
    "torch_max_memory_reserved_gib": recs[-1]["max_memory_reserved_gib"],
    "nvidia_smi_peak_used_gib": round(max(smi) / 1024, 2),
    "nvidia_smi_samples": len(smi),
}
json.dump(s, open(f"{out}/summary-{tag}.json", "w"), indent=2)
print(json.dumps(s))
EOF
}

run 480x640 $S/bwm-ref/demo
run 480x640-eager $S/bwm-ref/demo --enforce-eager
run 672x896 $D
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > $OUT/nvidia-smi.txt
