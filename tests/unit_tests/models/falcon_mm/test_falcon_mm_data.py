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

"""Unit tests for the Falcon-MM Energon task-encoder config."""

from megatron.bridge.data.builders.energon import QwenVLEnergonTaskEncoderConfig
from megatron.bridge.models.falcon_mm.data import (
    FalconEnergonTaskEncoderConfig,
    FalconTaskEncoder,
)
from megatron.bridge.models.qwen_vl.data.energon import QwenVLTaskEncoder


class TestFalconEnergonTaskEncoderConfig:
    def test_passes_qwen_isinstance_gate(self):
        """The packing gate in data/builders/energon.py matches the Qwen config.

        The Falcon config must remain a subclass so Energon-native packing is
        accepted with no shared-code edits.
        """
        config = FalconEnergonTaskEncoderConfig(hf_processor_path="Qwen/Qwen3.5-2B")
        assert isinstance(config, QwenVLEnergonTaskEncoderConfig)

    def test_inherits_qwen_defaults(self):
        config = FalconEnergonTaskEncoderConfig(hf_processor_path="Qwen/Qwen3.5-2B")
        assert config.spatial_merge_size == 2
        assert config.max_num_images == 10
        assert config.max_visual_tokens == 16384

    def test_falcon_task_encoder_is_qwen_task_encoder(self):
        assert issubclass(FalconTaskEncoder, QwenVLTaskEncoder)
