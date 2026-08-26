#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Falcon-MM 2B (encoder-free, Qwen3.5-2B trunk) dev run with Energon-native THD packing.
#
# Default data is the pretraining interleaved web-document mixture:
# MINT-1T-ArXiv image-text + fineweb text-onl mixed together.
# Check the folder's README.md for detail.
# The FalconTaskEncoder consumes both as raw documents, no chat template.
#
# Point ENERGON_PATH at any indexed dataset root / MetadatasetV2 YAML to override.
# The epochized mixture loops indefinitely (Energon default repeat=True)
# with exact per-epoch ratios from the YAML repetitions.
set -euo pipefail

export OMP_NUM_THREADS=8
WORKSPACE=${WORKSPACE:-/opt/Megatron-Bridge/workspace}
ENERGON_PATH="${ENERGON_PATH:-${WORKSPACE}/data/falcon_pretrain_mixture.yaml}"
PRETRAINED_CHECKPOINT=${PRETRAINED_CHECKPOINT:-${WORKSPACE}/models/Qwen/Qwen3.5-2B/}
OUTPUT_DIR=${OUTPUT_DIR:-${WORKSPACE}/results/falcon_mm_2b_sft}
RECIPE=${RECIPE:-falcon_mm_2b_sft_1gpu_h100_bf16_config}
NUM_GPUS=${NUM_GPUS:-8}
SEQ_LENGTH=${SEQ_LENGTH:-8192}
TRAIN_ITERS=${TRAIN_ITERS:-1000}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-16}
EVAL_ITERS=${EVAL_ITERS:-1}
EVAL_INTERVAL=${EVAL_INTERVAL:-20000}
PACKING_BUFFER_SIZE=${PACKING_BUFFER_SIZE:-16}

# Set WANDB_API_KEY to log online, or keep the default disabled mode.
export WANDB_MODE=${WANDB_MODE:-online}
WANDB_PROJECT=${WANDB_PROJECT:-megatron-bridge}

if [[ ! -e "${ENERGON_PATH}" ]]; then
  echo "ENERGON_PATH must be an indexed Energon dataset directory or a Metadataset YAML: ${ENERGON_PATH}" >&2
  exit 1
fi
MBS=1
# Energon native packing requires physical micro_batch_size=1. PP and VPP are
# supported (packs forced fixed-width; every stage pulls identical packs; VPP
# auto-disables batched p2p which mispairs traffic under the interleaved
# schedule). PP must divide NUM_GPUS.
PP=${PP:-1}
VPP=${VPP:-}
VPP_ARG=()
if [[ -n "${VPP}" ]]; then
  VPP_ARG=(model.virtual_pipeline_model_parallel_size="${VPP}")
fi
uv run python -m torch.distributed.run --standalone --nproc_per_node="${NUM_GPUS}" \
  scripts/training/run_recipe.py \
  --recipe "${RECIPE}" \
  --step_func vlm_step \
  checkpoint.pretrained_checkpoint="${PRETRAINED_CHECKPOINT}" \
  checkpoint.load=null \
  checkpoint.save="${OUTPUT_DIR}/checkpoints" \
  train.train_iters="${TRAIN_ITERS}" \
  train.global_batch_size="${GLOBAL_BATCH_SIZE}" \
  train.micro_batch_size="${MBS}" \
  validation.eval_interval="${EVAL_INTERVAL}" \
  validation.eval_iters="${EVAL_ITERS}" \
  validation.eval_micro_batch_size="${MBS}" \
  model.seq_length="${SEQ_LENGTH}" \
  model.freeze_language_model=False \
  model.pipeline_model_parallel_size="${PP}" \
  "${VPP_ARG[@]}" \
  dataset.path="${ENERGON_PATH}" \
  dataset.micro_batch_size="${MBS}" \
  dataset.seq_length="${SEQ_LENGTH}" \
  dataset.packing_buffer_size="${PACKING_BUFFER_SIZE}" \
  dataset.num_workers=8 \
  optimizer.lr=1e-4 \
  scheduler.lr_warmup_iters=100 \
  logger.log_interval=1 \
  logger.log_device_memory_used=True \
  logger.wandb_project="${WANDB_PROJECT}" \
  "$@"
