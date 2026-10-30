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

set -euo pipefail

# QwenVLEnergonTaskEncoderConfig expects ChatML conversations and media.
# For an 8-GPU mock run, generate a compatible indexed dataset with
# tutorials/data/energon/prepare_example_data.py --train-samples 1024.
: "${ENERGON_PATH:?Set ENERGON_PATH to an indexed Qwen ChatMLWebdataset root}"
: "${WANDB_PROJECT:?Set WANDB_PROJECT for throughput logging}"

GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-64}
TRAIN_ITERS=${TRAIN_ITERS:-50}
PACKING_BUFFER_SIZE=${PACKING_BUFFER_SIZE:-16}

uv run python -m torch.distributed.run --standalone --nproc_per_node=8 \
  scripts/training/run_recipe.py \
  --recipe qwen3_vl_30b_a3b_pretrain_8gpu_b200_fp8mx_energon_config \
  --mode pretrain \
  --step_func vlm_step \
  dataset.path="${ENERGON_PATH}" \
  dataset.packing_buffer_size="${PACKING_BUFFER_SIZE}" \
  train.global_batch_size="${GLOBAL_BATCH_SIZE}" \
  dataset.seq_length=12288 \
  train.train_iters="${TRAIN_ITERS}" \
  scheduler.lr_decay_iters="${TRAIN_ITERS}" \
  logger.wandb_project="${WANDB_PROJECT}" \
  logger.wandb_exp_name=qwen3-vl-30b-b200-fp8mx-energon-8k \
  logger.log_interval=1 \
  logger.log_throughput_to_tensorboard=True \
  logger.throughput_window_size=10 \
  logger.log_timers_to_tensorboard=True \
  logger.log_device_memory_used=True
