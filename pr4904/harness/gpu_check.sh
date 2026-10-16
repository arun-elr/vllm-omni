#!/bin/bash
# Post-fix GPU check (PR #4904 review 3), from the scratch dir: the documented
# 143-frame command on the new code (default compile and --enforce-eager),
# compared frame by frame to the earlier runs on the old code, plus an
# end-to-end output_type="latent" request.
set -euo pipefail
S=$(pwd)
NEW=$S/vllm-omni-new
OMNI=$S/venv-omni/bin/python
OUT=$S/check
mkdir -p $OUT
exec 9>/tmp/h200.lock
flock 9

run() {
  local tag=$1; shift
  (cd $NEW && PYTHONPATH=$NEW /usr/bin/time -v $OMNI examples/offline_inference/bwm/bwm_world_model.py \
      --model $S/models/BWM --episode-dir $S/bwm-ref/demo --episode 0 --output $OUT/bwm_rollout_$tag.mp4 "$@") \
      > $OUT/log-$tag.log 2>&1
}

[ -f $OUT/bwm_rollout_480x640.mp4 ] || run 480x640
[ -f $OUT/bwm_rollout_480x640-eager.mp4 ] || run 480x640-eager --enforce-eager

[ -f $OUT/latent_e2e.json ] || (cd $NEW && PYTHONPATH=$NEW $OMNI $S/harness/latent_e2e.py $S/models/BWM $S/bwm-ref/demo \
    $NEW/examples/offline_inference/bwm/bwm_world_model.py $OUT/latent_e2e.json) > $OUT/log-latent-e2e.log 2>&1

$OMNI - $S <<'EOF'
import hashlib, json, sys
import imageio.v2 as imageio
import numpy as np
S = sys.argv[1]
res = {}
for tag in ("480x640", "480x640-eager"):
    old, new = f"{S}/mem/bwm_rollout_{tag}.mp4", f"{S}/check/bwm_rollout_{tag}.mp4"
    a = np.stack([f for f in imageio.get_reader(old)])
    b = np.stack([f for f in imageio.get_reader(new)])
    same_shape = a.shape == b.shape
    d = np.abs(a.astype(np.int16) - b.astype(np.int16)) if same_shape else None
    res[tag] = {
        "old_frames": int(a.shape[0]), "new_frames": int(b.shape[0]),
        "hw": list(b.shape[1:3]),
        "decoded_frames_identical": bool(same_shape and not d.any()),
        "max_abs_diff": int(d.max()) if same_shape else None,
        "mp4_sha256_equal": hashlib.sha256(open(old, "rb").read()).hexdigest()
        == hashlib.sha256(open(new, "rb").read()).hexdigest(),
    }
res["latent_e2e"] = json.load(open(f"{S}/check/latent_e2e.json"))
json.dump(res, open(f"{S}/check/summary.json", "w"), indent=2)
print(json.dumps(res, indent=2))
EOF
