# Copyright 2023 The HuggingFace Team. All rights reserved.
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
from typing import TYPE_CHECKING

from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_tf_available,
    is_torch_available,
)


_import_structure = {"configuration_dimbart": ["DIMBART_PRETRAINED_CONFIG_ARCHIVE_MAP", "DiMBartConfig", "DiMBartOnnxConfig"]}

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_dimbart"] = [
        "DIMBART_PRETRAINED_MODEL_ARCHIVE_LIST",
        "DiMBartForCausalLM",
        "DiMBartForConditionalGeneration",
        "DiMBartForQuestionAnswering",
        "DiMBartForSequenceClassification",
        "DiMBartModel",
        "DiMBartPreTrainedModel",
    ]

try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_tf_dimbart"] = [
        "TFDiMBartForConditionalGeneration",
        "TFDiMBartModel",
        "TFDiMBartPreTrainedModel",
    ]

try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_flax_dimbart"] = [
        "FlaxDiMBartForConditionalGeneration",
        "FlaxDiMBartForQuestionAnswering",
        "FlaxDiMBartForSequenceClassification",
        "FlaxDiMBartModel",
        "FlaxDiMBartPreTrainedModel",
    ]


if TYPE_CHECKING:
    from .configuration_dimbart import DIMBART_PRETRAINED_CONFIG_ARCHIVE_MAP, DiMBartConfig, DiMBartOnnxConfig

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_dimbart import (
            DIMBART_PRETRAINED_MODEL_ARCHIVE_LIST,
            DiMBartForCausalLM,
            DiMBartForConditionalGeneration,
            DiMBartForQuestionAnswering,
            DiMBartForSequenceClassification,
            DiMBartModel,
            DiMBartPreTrainedModel,
        )

    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_tf_dimbart import TFDiMBartForConditionalGeneration, TFDiMBartModel, TFDiMBartPreTrainedModel

    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_flax_dimbart import (
            FlaxDiMBartForConditionalGeneration,
            FlaxDiMBartForQuestionAnswering,
            FlaxDiMBartForSequenceClassification,
            FlaxDiMBartModel,
            FlaxDiMBartPreTrainedModel,
        )

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
