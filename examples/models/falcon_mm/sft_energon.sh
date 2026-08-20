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

# Falcon-MM 2B (encoder-free, Qwen3.5-2B trunk) dev run with Energon-native
# THD packing. The default recipe is the PP=2 bring-up variant; use
# RECIPE=falcon_mm_2b_sft_1gpu_h100_bf16_config NUM_GPUS=1 for the PP=1
# loss-parity baseline (energon-packing-pp-enablement.md, step 2).
#
# Prepare ENERGON_PATH with tutorials/data/energon/README.md or the Mantis
# converter before launching. An indexed dataset root or a MetadatasetV2 YAML
# mixture path both work (Energon 7 metadataset support).
#
# Tiny mock roots (few samples) must use finite Energon repetitions: indefinite
# repeat asserts on empty dataloader workers ("Cannot repeat empty dataset
# indefinitely"), which commonly hits at the first eval with a tiny val split.
set -euo pipefail

ENERGON_PATH="${ENERGON_PATH:-/opt/Megatron-Bridge/workspace/energon-mock}"

WORKSPACE=${WORKSPACE:-/opt/Megatron-Bridge/workspace}
PRETRAINED_CHECKPOINT=${PRETRAINED_CHECKPOINT:-${WORKSPACE}/models/Qwen/Qwen3.5-2B/}
OUTPUT_DIR=${OUTPUT_DIR:-${WORKSPACE}/results/falcon_mm_2b_sft}
RECIPE=${RECIPE:-falcon_mm_2b_sft_1gpu_h100_bf16_config}
NUM_GPUS=${NUM_GPUS:-2}
SEQ_LENGTH=${SEQ_LENGTH:-4096}
TRAIN_ITERS=${TRAIN_ITERS:-100}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-8}
EVAL_ITERS=${EVAL_ITERS:-1}
EVAL_INTERVAL=${EVAL_INTERVAL:-20}
PACKING_BUFFER_SIZE=${PACKING_BUFFER_SIZE:-4}
DATASET_REPEATS=${DATASET_REPEATS:-100}
WANDB_PROJECT=${WANDB_PROJECT:-megatron-bridge}

# Set WANDB_API_KEY to log online, or keep the default disabled mode.
export WANDB_MODE=${WANDB_MODE:-online}

DATASET_PATH="${ENERGON_PATH}"
DATASET_EXTRA_ARGS=()
if [[ -d "${ENERGON_PATH}" ]]; then
  # Wrap the indexed root in a MetadatasetV2 blend_epochized mixture so each
  # sample is seen DATASET_REPEATS times per epoch. repeat=false disables the
  # outer indefinite RepeatDataset that asserts on empty workers.
  ENERGON_ABS="$(cd "${ENERGON_PATH}" && pwd)"
  METADATASET_PATH="${OUTPUT_DIR}/energon_metadataset_repeats.yaml"
  mkdir -p "${OUTPUT_DIR}"
  cat > "${METADATASET_PATH}" <<EOF
__module__: megatron.energon
__class__: MetadatasetV2
splits:
  train:
    blend_epochized:
      - path: ${ENERGON_ABS}
        repetitions: ${DATASET_REPEATS}
  val:
    blend_epochized:
      - path: ${ENERGON_ABS}
        repetitions: ${DATASET_REPEATS}
EOF
  DATASET_PATH="${METADATASET_PATH}"
  DATASET_EXTRA_ARGS+=(+dataset.dataset_kwargs.repeat=false)
elif [[ ! -f "${ENERGON_PATH}" ]]; then
  echo "ENERGON_PATH must be an indexed Energon dataset directory or a Metadataset YAML: ${ENERGON_PATH}" >&2
  exit 1
fi

# Energon native packing requires physical micro_batch_size=1; PP comes from
# the recipe (override with model.pipeline_model_parallel_size=N if needed).
uv run python -m torch.distributed.run --standalone --nproc_per_node="${NUM_GPUS}" \
  scripts/training/run_recipe.py \
  --recipe "${RECIPE}" \
  --step_func vlm_step \
  checkpoint.pretrained_checkpoint="${PRETRAINED_CHECKPOINT}" \
  checkpoint.load=null \
  checkpoint.save="${OUTPUT_DIR}/checkpoints" \
  train.train_iters="${TRAIN_ITERS}" \
  train.global_batch_size="${GLOBAL_BATCH_SIZE}" \
  train.micro_batch_size=1 \
  validation.eval_interval="${EVAL_INTERVAL}" \
  validation.eval_iters="${EVAL_ITERS}" \
  validation.eval_micro_batch_size=1 \
  model.seq_length="${SEQ_LENGTH}" \
  dataset.path="${DATASET_PATH}" \
  dataset.seq_length="${SEQ_LENGTH}" \
  dataset.micro_batch_size=1 \
  model.pipeline_model_parallel_size=1 \
  model.moe_token_dispatcher_type=all2all \
  dataset.packing_buffer_size="${PACKING_BUFFER_SIZE}" \
  "${DATASET_EXTRA_ARGS[@]}" \
  logger.log_interval=1 \
  logger.log_device_memory_used=True \
  logger.wandb_project="${WANDB_PROJECT}" \
  "$@"
