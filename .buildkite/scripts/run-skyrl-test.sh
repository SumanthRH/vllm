#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Setup and run SkyRL GSM8K colocated GRPO training integration test
# This script validates that vLLM works correctly as an inference backend for SkyRL

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SKYRL_REPO="https://github.com/SkyworkAI/skyrl.git"
SKYRL_DIR="${REPO_ROOT}/skyrl"

if command -v rocm-smi &> /dev/null || command -v rocminfo &> /dev/null; then
    echo "AMD GPU detected. SkyRL currently only supports NVIDIA. Skipping..."
    exit 0
fi

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
    source "$HOME"/.local/bin/env
fi

# Clone SkyRL repository
SKYRL_BRANCH="main"
echo "Cloning SkyRL repository at branch: ${SKYRL_BRANCH}..."
git clone --branch "${SKYRL_BRANCH}" --single-branch "${SKYRL_REPO}" "${SKYRL_DIR}"
cd "${SKYRL_DIR}"

echo "Setting up UV project environment..."
export UV_PROJECT_ENVIRONMENT=/usr/local
if [ ! -f /usr/local/bin/python ]; then
    ln -s /usr/bin/python3 /usr/local/bin/python
fi

# Remove vllm pin from pyproject.toml so we test against vLLM main
echo "Removing vllm pin from pyproject.toml..."
sed -i '/vllm==/d' pyproject.toml

# Sync SkyRL dependencies
echo "Installing SkyRL dependencies..."
uv sync --inexact --extra fsdp --extra skyrl-train

# Verify installation
echo "Verifying installations..."
python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"
python -c "import skyrl; print('SkyRL imported successfully')"

# Prepare GSM8K dataset
echo "Preparing GSM8K dataset..."
python examples/data_preprocess/gsm8k.py

# Run training
echo "Running SkyRL GSM8K colocated GRPO training..."
export WANDB_MODE=offline
export NUM_GPUS=2

RUN_NAME="gsm8k_ci_$(date +%s)"

bash examples/train/gsm8k/run_gsm8k.sh \
    trainer.epochs=1 \
    trainer.eval_before_train=true \
    trainer.micro_forward_batch_size_per_gpu=16 \
    trainer.micro_train_batch_size_per_gpu=16 \
    trainer.project_name=\"gsm8k_ci\" \
    trainer.run_name=\"$RUN_NAME\"

echo "Training completed. Validating metrics..."

# Validate metrics from wandb offline run directory
python3 -c "
import json
import glob
import sys

# Find the wandb summary file from the latest offline run
summary_patterns = [
    'wandb/latest-run/files/wandb-summary.json',
    'wandb/offline-run-*/files/wandb-summary.json',
]

summary_file = None
for pattern in summary_patterns:
    matches = sorted(glob.glob(pattern))
    if matches:
        summary_file = matches[-1]
        break

if summary_file is None:
    print('ERROR: Could not find wandb-summary.json')
    sys.exit(1)

print(f'Found summary file: {summary_file}')

with open(summary_file) as f:
    summary = json.load(f)

print(f'Summary contents: {json.dumps(summary, indent=2)}')

# Define thresholds (from gsm8k_colocate.sh)
checks = [
    ('eval/all/avg_score', '>=', 0.69),
    ('loss/avg_final_rewards', '>=', 0.69),
    ('generate/avg_num_tokens', '<=', 232),
    ('policy/rollout_train_logprobs_abs_diff_mean', '<=', 0.0104),
]

failed = False
for metric_key, op, threshold in checks:
    # Handle nested keys (e.g., 'eval/all/avg_score')
    value = summary.get(metric_key)
    if value is None:
        print(f'WARNING: Metric {metric_key} not found in summary')
        failed = True
        continue

    if op == '>=' and value < threshold:
        print(f'FAIL: {metric_key} = {value} (expected >= {threshold})')
        failed = True
    elif op == '<=' and value > threshold:
        print(f'FAIL: {metric_key} = {value} (expected <= {threshold})')
        failed = True
    else:
        print(f'PASS: {metric_key} = {value} ({op} {threshold})')

if failed:
    print('Metric validation FAILED')
    sys.exit(1)

print('All metric validations PASSED')
"

echo "SkyRL integration test completed successfully!"
