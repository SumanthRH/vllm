#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Setup script for Prime-RL integration tests
# This script prepares the environment for running Prime-RL tests with nightly vLLM

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SKYRL_REPO="https://github.com/NovaSky-AI/SkyRL.git"
SKYRL_DIR="${REPO_ROOT}/skyrl"

echo "Setting up SkyRL integration test environment..."

# Clean up any existing SkyRL directory
if [ -d "${SKYRL_DIR}" ]; then
    echo "Removing existing SkyRL directory..."
    rm -rf "${SKYRL_DIR}"
fi

# Install UV if not available
if ! command -v uv &> /dev/null; then
    echo "Installing UV package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env
fi

# Clone SkyRL repository at specific branch for reproducible tests
SKYRL_BRANCH="vllm-integration"
echo "Cloning SkyRL repository at branch: ${SKYRL_BRANCH}..."
git clone --branch "${SKYRL_BRANCH}" --single-branch "${SKYRL_REPO}" "${SKYRL_DIR}"
cd "${SKYRL_DIR}/skyrl-train"

# echo "Setting up UV project environment..."
# export UV_PROJECT_ENVIRONMENT=/usr/local
# ln -s /usr/bin/python3 /usr/local/bin/python

# # Ensure SkyRL is compatible with the current vLLM version
echo "Ensuring SkyRL is compatible with vLLM nightly..."
uv run --extra vllm --with "vllm@file://${REPO_ROOT}" -- python -c "import skyrl_train"

echo "SkyRL installation successful!"


echo "Running SkyRL integration tests..."

use_flash_attn=true
use_sample_packing=true
flash_attn_works=$(uv run --extra vllm --with "vllm@file://${REPO_ROOT}" -- python -c "import flash_attn" || echo "false")
if [ "$flash_attn_works" == "false" ]; then
    echo "Flash attention installation failed, disabling flash attention for the integration test"
    use_flash_attn=false
    use_sample_packing=false
fi

uv run examples/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
LOGGER=console bash examples/gsm8k/run_gsm8k.sh \
  trainer.policy.model.path="Qwen/Qwen2.5-0.5B-Instruct" \
  trainer.epochs=1 \
  trainer.eval_before_train=false \
  trainer.micro_forward_batch_size_per_gpu=16 \
  trainer.micro_train_batch_size_per_gpu=16 \
  trainer.ckpt_interval=-1 \
  trainer.eval_interval=-1 \
  trainer.flash_attn=$use_flash_attn \
  trainer.use_sample_packing=$use_sample_packing \

echo "SkyRL integration tests completed!"
