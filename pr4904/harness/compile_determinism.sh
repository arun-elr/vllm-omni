#!/bin/bash
# Is the default (torch.compile) rollout deterministic run to run? Rerun the
# documented command on the old code (a548404) and compare the old/old and
# old/new pairs by PSNR, from the scratch dir.
set -euo pipefail
S=$(pwd)
OMNI=$S/venv-omni/bin/python
OUT=$S/check
exec 9>/tmp/h200.lock
flock 9
[ -f $OUT/bwm_rollout_480x640-oldrerun.mp4 ] || (cd $S/vllm-omni-a548404 && PYTHONPATH=$S/vllm-omni-a548404 \
    $OMNI examples/offline_inference/bwm/bwm_world_model.py --model $S/models/BWM \
    --episode-dir $S/bwm-ref/demo --episode 0 --output $OUT/bwm_rollout_480x640-oldrerun.mp4) \
    > $OUT/log-480x640-oldrerun.log 2>&1

$OMNI - $S <<'EOF'
import json, sys
import imageio.v2 as imageio
import numpy as np
S = sys.argv[1]
def load(p):
    return np.stack([f for f in imageio.get_reader(p)]).astype(np.float64)
def psnr(a, b):
    mse = ((a - b) ** 2).mean()
    return float("inf") if mse == 0 else float(10 * np.log10(255.0**2 / mse))
old = load(f"{S}/mem/bwm_rollout_480x640.mp4")
old2 = load(f"{S}/check/bwm_rollout_480x640-oldrerun.mp4")
new = load(f"{S}/check/bwm_rollout_480x640.mp4")
res = {
    "compiled_old_vs_old_rerun": {"psnr_db": psnr(old, old2), "max_abs": int(np.abs(old - old2).max())},
    "compiled_old_vs_new": {"psnr_db": psnr(old, new), "max_abs": int(np.abs(old - new).max())},
    "compiled_old_rerun_vs_new": {"psnr_db": psnr(old2, new), "max_abs": int(np.abs(old2 - new).max())},
}
json.dump(res, open(f"{S}/check/compile_determinism.json", "w"), indent=2)
print(json.dumps(res, indent=2))
EOF
