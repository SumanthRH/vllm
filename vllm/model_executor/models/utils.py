# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, overload

import torch
import torch.nn as nn
from torch.func import functional_call
from transformers import PretrainedConfig
from typing_extensions import deprecated

from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
)
from vllm.model_executor.model_loader.online_quantization import (
    support_quantized_model_reload_from_hp_weights,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import supports_any_eagle
from vllm.multimodal import NestedTensors
from vllm.sequence import IntermediateTensors
from vllm.utils.math_utils import cdiv
from vllm.utils.platform_utils import (
    is_pin_memory_available,
    is_uva_available,
)
from vllm.utils.torch_utils import (
    direct_register_custom_op,
    get_cuda_view_from_cpu_tensor,
)

# Chunk x along the num_tokens axis for sequence parallelism
# NOTE: This is wrapped in a torch custom op to work around the following issue:
# The output tensor can have a sequence length 0 at small input sequence lengths
# even though we explicitly pad to avoid this.
def sequence_parallel_chunk(x: torch.Tensor) -> torch.Tensor:
    return torch.ops.vllm.sequence_parallel_chunk_impl(x)


def sequence_parallel_chunk_impl(x: torch.Tensor) -> torch.Tensor:
    tp_size = get_tensor_model_parallel_world_size()
    tp_rank = get_tensor_model_parallel_rank()

    # all_gather needs the sequence length to be divisible by tp_size
    seq_len = x.size(0)
    remainder = seq_len % tp_size
    if remainder != 0:
        pad_len = tp_size - remainder
        y = nn.functional.pad(x, (0, 0, 0, pad_len))
    else:
        y = x

    chunk = y.shape[0] // tp_size
    start = tp_rank * chunk
    return torch.narrow(y, 0, start, chunk)


def sequence_parallel_chunk_impl_fake(x: torch.Tensor) -> torch.Tensor:
    tp_size = get_tensor_model_parallel_world_size()
    seq_len = cdiv(x.size(0), tp_size)
    shape = list(x.shape)
    shape[0] = seq_len
    out = torch.empty(shape, dtype=x.dtype, device=x.device)
    return out


direct_register_custom_op(
    op_name="sequence_parallel_chunk_impl",
    op_func=sequence_parallel_chunk_impl,
    fake_impl=sequence_parallel_chunk_impl_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


def process_eagle_weight(
    model: nn.Module,
    name: str,
) -> None:
    """
    Update EAGLE model flags based on loaded weight name.
    This should be called during weight loading to detect if a model
    has its own lm_head or embed_tokens weight.
    Args:
        model: The model instance (must support EAGLE)
        name: The name of the weight to process
    """
    if not supports_any_eagle(model):
        return

    # To prevent overriding with target model's layers
    if "lm_head" in name:
        model.has_own_lm_head = True
    if "embed_tokens" in name:
        model.has_own_embed_tokens = True
