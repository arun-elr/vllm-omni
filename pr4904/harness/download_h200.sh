#!/bin/bash
# Pinned weights. HF revisions recorded in env/revisions.txt.
set -euxo pipefail
export PATH="$HOME/.local/bin:$PATH"
S=$HOME/scratch-pr4904
mkdir -p $S/weights
hf download BLM-Lab/Boundless-World-Model step-12000.safetensors config.json --revision 738a8d3c008e637b8b1b18d5e98a82f6de9c04aa --local-dir $S/weights/BLM
hf download Wan-AI/Wan2.2-TI2V-5B --revision 921dbaf3f1674a56f47e83fb80a34bac8a8f203e --local-dir $S/weights/Wan2.2-TI2V-5B \
  diffusion_pytorch_model-00001-of-00003.safetensors diffusion_pytorch_model-00002-of-00003.safetensors \
  diffusion_pytorch_model-00003-of-00003.safetensors diffusion_pytorch_model.safetensors.index.json Wan2.2_VAE.pth config.json
hf download Wan-AI/Wan2.2-TI2V-5B-Diffusers --revision b8fff7315c768468a5333511427288870b2e9635 --include 'transformer/*' --include 'vae/*' --include model_index.json
echo DL_DONE
