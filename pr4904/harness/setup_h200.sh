#!/bin/bash
# Throwaway envs + pinned weights for the PR #4904 parity run. Scratch dir only.
set -euxo pipefail
export PATH="$HOME/.local/bin:$PATH"
S=$HOME/scratch-pr4904
cd $S

# PR side: vllm 0.24.0 (pinned in the PR head's install docs) + vllm-omni at a54840407, editable.
if [ ! -x venv-omni/bin/python ]; then
  uv venv -q venv-omni --python 3.12
  VIRTUAL_ENV=$S/venv-omni uv pip install -q vllm==0.24.0 --torch-backend=auto
  VIRTUAL_ENV=$S/venv-omni SETUPTOOLS_SCM_PRETEND_VERSION=0.24.0rc2.dev47+ga54840407 uv pip install -q -e $S/vllm-omni-a548404
  VIRTUAL_ENV=$S/venv-omni uv pip install -q imageio imageio-ffmpeg pyarrow pytest scikit-image matplotlib
fi

# Reference side: exactly the reference README install (torch 2.8.0 cu128, diffsynth 2.0.11, requirements.txt).
if [ ! -x venv-ref/bin/python ]; then
  uv venv -q venv-ref --python 3.12
  VIRTUAL_ENV=$S/venv-ref uv pip install -q torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
  VIRTUAL_ENV=$S/venv-ref uv pip install -q diffsynth==2.0.11
  VIRTUAL_ENV=$S/venv-ref uv pip install -q -r $S/bwm-ref/requirements.txt
  VIRTUAL_ENV=$S/venv-ref uv pip install -q pyyaml
fi
echo ENVS_DONE
