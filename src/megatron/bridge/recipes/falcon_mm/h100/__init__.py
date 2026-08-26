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

from megatron.bridge.recipes.falcon_mm.h100.falcon_mm import *  # noqa: F403


__all__ = [
    "falcon_mm_2b_sft_1gpu_h100_bf16_config",
    "falcon_mm_2b_sft_2gpu_h100_bf16_config",
    "falcon_mm_2b_moe_pretrain_4gpu_h100_bf16_config",
    "falcon_mm_2b_moe_pretrain_8gpu_h100_bf16_config",
    "falcon_mm_35b_a3b_sft_8gpu_h100_bf16_config",
    "falcon_mm_35b_a3b_sft_16gpu_h100_bf16_config",
]
