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

# Falcon-MM dev run with Energon-native THD packing.
# Default data is the pretraining interleaved web-document mixture:
# MINT-1T-ArXiv image-text + fineweb text-only mixed together.
# Check the folder's README.md for detail.
# The FalconTaskEncoder consumes both as raw documents, no chat template.
#
# Point ENERGON_PATH at any indexed dataset root / MetadatasetV2 YAML to override.
# The epochized mixture loops indefinitely (Energon default repeat=True)
# with exact per-epoch ratios from the YAML repetitions.
#
# Throughput sibling of the Qwen3-30B-A3B 8-GPU MXFP8 recipe (GBS=512, seq=4096):
#   RECIPE=falcon_deepseekv4_moe_35b_a3b_pretrain_8gpu_b200_fp8mx_config \
#   SEQ_LENGTH=4096 GLOBAL_BATCH_SIZE=512 ./examples/models/falcon_mm/thd_energon.sh
set -euo pipefail

export OMP_NUM_THREADS=8
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export CUDA_HOME=/usr/local/cuda
export CUDA_PATH=/usr/local/cuda
export CPATH=/usr/local/cuda/include${CPATH:+:$CPATH}
export CPATH="${CUDA_HOME}/targets/x86_64-linux/include/cccl${CPATH:+:$CPATH}"

WORKSPACE=${WORKSPACE:-/mnt/checkpoints/phuc}
ENERGON_PATH=${ENERGON_PATH:-~/phuc/Megatron-Bridge/examples/models/falcon_mm/pretrain_mixture.yaml}
PRETRAINED_CHECKPOINT=${PRETRAINED_CHECKPOINT:-null}
LOAD_CHECKPOINT=${LOAD_CHECKPOINT:-null}
OUTPUT_DIR=${OUTPUT_DIR:-${WORKSPACE}/falcon_mm_35b_pretrain}
RECIPE=${RECIPE:-falcon_deepseekv4_1b_pretrain_8gpu_b200_bf16_config}
NUM_GPUS=${NUM_GPUS:-8}
LAST_RANK=$((NUM_GPUS-1))
SEQ_LENGTH=${SEQ_LENGTH:-4096}
TRAIN_ITERS=${TRAIN_ITERS:-20000}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-$NUM_GPUS}
EVAL_ITERS=${EVAL_ITERS:-10}
EVAL_INTERVAL=${EVAL_INTERVAL:-1000}
PACKING_BUFFER_SIZE=${PACKING_BUFFER_SIZE:-64}
LOGDIR=${LOGDIR:-LOGS/}
# Set WANDB_API_KEY to log online, or keep the default disabled mode.
export WANDB_MODE=${WANDB_MODE:-online}
WANDB_PROJECT=${WANDB_PROJECT:-megatron-bridge}

if [[ ! -e "${ENERGON_PATH}" ]]; then
  echo "ENERGON_PATH must be an indexed Energon dataset directory or a Metadataset YAML: ${ENERGON_PATH}" >&2
  exit 1
fi
# micro_batch_size is fixed at 1 in Falcon Energon recipes (online THD packing).
# PP and VPP are supported (packs forced fixed-width; every stage pulls identical
# packs; VPP auto-disables batched p2p which mispairs traffic under the
# interleaved schedule). PP must divide NUM_GPUS.
PP=${PP:-1}
VPP=${VPP:-}
VPP_ARG=()
if [[ -n "${VPP}" ]]; then
  VPP_ARG=(model.virtual_pipeline_model_parallel_size="${VPP}")
fi
uv run python -m torch.distributed.run --standalone --nproc_per_node="${NUM_GPUS}" \
  --log-dir "$LOGDIR" --tee "$LAST_RANK:3" \
  scripts/training/run_recipe.py \
  --recipe "${RECIPE}" \
  --step_func vlm_step \
  checkpoint.pretrained_checkpoint="${PRETRAINED_CHECKPOINT}" \
  checkpoint.load="${LOAD_CHECKPOINT}" \
  checkpoint.save="${OUTPUT_DIR}/checkpoints" \
  train.train_iters="${TRAIN_ITERS}" \
  train.global_batch_size="${GLOBAL_BATCH_SIZE}" \
  validation.eval_interval="${EVAL_INTERVAL}" \
  validation.eval_iters="${EVAL_ITERS}" \
  model.seq_length="${SEQ_LENGTH}" \
  model.pipeline_model_parallel_size="${PP}" \
  "${VPP_ARG[@]}" \
  dataset.path="${ENERGON_PATH}" \
  dataset.seq_length="${SEQ_LENGTH}" \
  dataset.packing_buffer_size="${PACKING_BUFFER_SIZE}" \
  dataset.num_workers=16 \
  dataset.num_val_workers=4 \
  optimizer.lr=1e-4 \
  scheduler.lr_warmup_iters=500 \
  logger.console_color=always \
  logger.log_interval=1 \
  logger.log_device_memory_used=True \
  logger.wandb_project="${WANDB_PROJECT}" \
  logger.timing_log_level=2 \
  logger.log_timers_to_tensorboard=True \
  "$@"
