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

from megatron.bridge.recipes.falcon_mm.b200.falcon_deepseekv4 import (
    falcon_deepseekv4_1b_pretrain_8gpu_b200_bf16_config,
    falcon_deepseekv4_3b_pretrain_8gpu_b200_bf16_config,
    falcon_deepseekv4_10b_pretrain_8gpu_b200_bf16_config,
    falcon_deepseekv4_35b_a3b_pretrain_8gpu_b200_bf16_config,
    falcon_deepseekv4_35b_a3b_pretrain_8gpu_b200_fp8mx_config,
    falcon_deepseekv4_flash_pretrain_8gpu_b200_bf16_config,
)


__all__ = [
    "falcon_deepseekv4_1b_pretrain_8gpu_b200_bf16_config",
    "falcon_deepseekv4_3b_pretrain_8gpu_b200_bf16_config",
    "falcon_deepseekv4_10b_pretrain_8gpu_b200_bf16_config",
    "falcon_deepseekv4_flash_pretrain_8gpu_b200_bf16_config",
    "falcon_deepseekv4_35b_a3b_pretrain_8gpu_b200_bf16_config",
    "falcon_deepseekv4_35b_a3b_pretrain_8gpu_b200_fp8mx_config",
]
