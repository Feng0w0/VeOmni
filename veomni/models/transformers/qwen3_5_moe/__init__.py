# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
from veomni.models.transformers.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeModel
from veomni.models.transformers.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeForConditionalGeneration
from veomni.models.transformers.qwen3_vl_moe.modeling_qwen3_vl_moe import apply_veomni_qwen3vlmoe_patch
from ...loader import MODELING_REGISTRY


@MODELING_REGISTRY.register("qwen3_5_moe")
def register_qwen3_5_moe_modeling(architecture: str):
    from .modeling_qwen3_5_moe import (
        Qwen3_5MoeForConditionalGeneration,
        Qwen3_5MoeModel,
        apply_veomni_qwen35moe_patch,
    )

    apply_veomni_qwen3vlmoe_patch()
    if "ForConditionalGeneration" in architecture:
        return Qwen3VLMoeForConditionalGeneration
    elif "Model" in architecture:
        return Qwen3VLMoeModel
    else:
        return Qwen3VLMoeForConditionalGeneration
