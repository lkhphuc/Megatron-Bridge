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

"""Unit tests for the Falcon-MM Llama3 provider (encoder-free dense GQA trunk)."""

from megatron.bridge.models.falcon_mm import (
    FALCON_LLAMA3_IMAGE_TOKEN_ID,
    FalconLlama3Model,
    FalconLlama3ModelProvider,
)


class TestFalconLlama3ModelProvider:
    def test_model_class_and_defaults(self):
        provider = FalconLlama3ModelProvider()
        assert provider.MODEL_CLASS is FalconLlama3Model
        assert provider.num_layers == 32
        assert provider.hidden_size == 4096
        assert provider.ffn_hidden_size == 14336
        assert provider.num_attention_heads == 32
        assert provider.num_query_groups == 8
        assert provider.vocab_size == 128256
        assert provider.rotary_base == 500000
        assert provider.position_embedding_type == "rope"
        assert provider.normalization == "RMSNorm"
        assert provider.gated_linear_unit is True
        assert provider.share_embeddings_and_output_weights is False
        assert provider.image_token_id == FALCON_LLAMA3_IMAGE_TOKEN_ID
        assert provider.patch_size == 32
        assert provider.spatial_merge_size == 1
        assert provider.temporal_patch_size == 1
        assert provider.patch_embed_num_layers == 3

    def test_patch_embed_config_matches_lm_hidden(self):
        provider = FalconLlama3ModelProvider(hidden_size=512)
        vision = provider._build_patch_embed_config()
        assert vision.hidden_size == 128
        assert vision.out_hidden_size == 512
        assert vision.patch_size == 32
        assert vision.spatial_merge_size == 1
        assert vision.temporal_patch_size == 1

    def test_custom_patch_embed_hidden_size_sets_intermediate_width(self):
        provider = FalconLlama3ModelProvider(hidden_size=520, patch_embed_hidden_size=111)
        vision = provider._build_patch_embed_config()
        assert vision.hidden_size == 111
        assert vision.pipeline_model_parallel_size == 1
