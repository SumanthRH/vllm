# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import warnings
from collections.abc import Callable
from dataclasses import InitVar, field
from importlib.util import find_spec
from typing import TYPE_CHECKING, Any, Literal, cast, get_args

import torch
from pydantic import ConfigDict, SkipValidation, field_validator, model_validator
from pydantic.dataclasses import dataclass
from safetensors.torch import _TYPES as _SAFETENSORS_TO_TORCH_DTYPE
from transformers.configuration_utils import ALLOWED_LAYER_TYPES

import vllm.envs as envs
from vllm.config.multimodal import MMCacheType, MMEncoderTPMode, MultiModalConfig
from vllm.config.pooler import PoolerConfig
from vllm.config.scheduler import RunnerType
from vllm.config.utils import config, getattr_iter
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.transformers_utils.config import (
    ConfigFormat,
    get_config,
    get_hf_image_processor_config,
    get_hf_text_config,
    get_pooling_config,
    get_sentence_transformer_tokenizer_config,
    is_encoder_decoder,
    try_get_dense_modules,
    try_get_generation_config,
    try_get_safetensors_metadata,
    try_get_tokenizer_config,
    uses_mrope,
    uses_xdrope_dim,
)
from vllm.transformers_utils.gguf_utils import (
    maybe_patch_hf_config_from_gguf,
)
from vllm.transformers_utils.runai_utils import ObjectStorageModel, is_runai_obj_uri
from vllm.transformers_utils.utils import (
    is_gguf,
    is_remote_gguf,
    maybe_model_redirect,
    split_remote_gguf,
)
from vllm.utils.import_utils import LazyLoader
from vllm.utils.torch_utils import common_broadcastable_dtype

if TYPE_CHECKING:
    from transformers import PretrainedConfig

    import vllm.model_executor.layers.quantization as me_quant
    import vllm.model_executor.models as me_models
    from vllm.attention.backends.registry import AttentionBackendEnum
    from vllm.config.load import LoadConfig
    from vllm.config.parallel import ParallelConfig
    from vllm.model_executor.layers.quantization import QuantizationMethods
    from vllm.v1.sample.logits_processor import LogitsProcessor
else:
    PretrainedConfig = Any

    AttentionBackendEnum = Any
    me_quant = LazyLoader(
        "model_executor", globals(), "vllm.model_executor.layers.quantization"
    )
    me_models = LazyLoader("model_executor", globals(), "vllm.model_executor.models")
    LoadConfig = Any
    ParallelConfig = Any
    QuantizationMethods = Any
    LogitsProcessor = Any

logger = init_logger(__name__)

RunnerOption = Literal["auto", RunnerType]
ConvertType = Literal["none", "embed", "classify", "reward"]
ConvertOption = Literal["auto", ConvertType]
TaskOption = Literal[
    "auto",
    "generate",
    "embedding",
    "embed",
    "classify",
    "score",
    "reward",
    "transcription",
    "draft",
]
TokenizerMode = Literal["auto", "hf", "slow", "mistral", "custom"]
ModelDType = Literal["auto", "half", "float16", "bfloat16", "float", "float32"]
LogprobsMode = Literal[
    "raw_logits", "raw_logprobs", "processed_logits", "processed_logprobs"
]
HfOverrides = dict[str, Any] | Callable[[PretrainedConfig], PretrainedConfig]
ModelImpl = Literal["auto", "vllm", "transformers", "terratorch"]
LayerBlockType = Literal["attention", "linear_attention", "mamba"]

_RUNNER_TASKS: dict[RunnerType, list[TaskOption]] = {
    "generate": ["generate", "transcription"],
    "pooling": ["embedding", "embed", "classify", "score", "reward"],
    "draft": ["draft"],
}

_RUNNER_CONVERTS: dict[RunnerType, list[ConvertType]] = {
    "generate": [],
    "pooling": ["embed", "classify", "reward"],
    "draft": [],
}


@config
@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class ModelConfig:
    """Configuration for the model."""

    model: str = "Qwen/Qwen3-0.6B"
    """Name or path of the Hugging Face model to use. It is also used as the
    content for `model_name` tag in metrics output when `served_model_name` is
    not specified."""
    runner: RunnerOption = "auto"
    """The type of model runner to use. Each vLLM instance only supports one
    model runner, even if the same model can be used for multiple types."""
    convert: ConvertOption = "auto"
    """Convert the model using adapters defined in
    [vllm.model_executor.models.adapters][]. The most common use case is to
    adapt a text generation model to be used for pooling tasks."""
    task: TaskOption | None = None
    """[DEPRECATED] The task to use the model for. If the model supports more
    than one model runner, this is used to select which model runner to run.

    Note that the model may support other tasks using the same model runner.
    """
    tokenizer: SkipValidation[str] = None  # type: ignore
    """Name or path of the Hugging Face tokenizer to use. If unspecified, model
    name or path will be used."""
    tokenizer_mode: TokenizerMode = "auto"
    """Tokenizer mode:\n
    - "auto" will use "hf" tokenizer if Mistral's tokenizer is not available.\n
    - "hf" will use the fast tokenizer if available.\n
    - "slow" will always use the slow tokenizer.\n
    - "mistral" will always use the tokenizer from `mistral_common`.\n
    - "custom" will use --tokenizer to select the preregistered tokenizer."""
    trust_remote_code: bool = False
    """Trust remote code (e.g., from HuggingFace) when downloading the model
    and tokenizer."""
    dtype: ModelDType | torch.dtype = "auto"
    """Data type for model weights and activations:\n
    - "auto" will use FP16 precision for FP32 and FP16 models, and BF16
    precision for BF16 models.\n
    - "half" for FP16. Recommended for AWQ quantization.\n
    - "float16" is the same as "half".\n
    - "bfloat16" for a balance between precision and range.\n
    - "float" is shorthand for FP32 precision.\n
    - "float32" for FP32 precision."""
    seed: int = 0
    """Random seed for reproducibility.

    We must set the global seed because otherwise,
    different tensor parallel workers would sample different tokens,
    leading to inconsistent results."""
    hf_config: PretrainedConfig = field(init=False)
    """The Hugging Face config of the model."""
    hf_text_config: PretrainedConfig = field(init=False)
    """The Hugging Face config of the text model (same as hf_config for text models)."""
    hf_config_path: str | None = None
    """Name or path of the Hugging Face config to use. If unspecified, model
    name or path will be used."""
    allowed_local_media_path: str = ""
    """Allowing API requests to read local images or videos from directories
    specified by the server file system. This is a security risk. Should only
    be enabled in trusted environments."""
    allowed_media_domains: list[str] | None = None
    """If set, only media URLs that belong to this domain can be used for
    multi-modal inputs. """
    revision: str | None = None

                and self.hf_text_config.kv_lora_rank is not None
            )
        return False

    def get_head_size(self) -> int:
        # TODO remove hard code
        if self.is_deepseek_mla:
            qk_rope_head_dim = getattr(self.hf_text_config, "qk_rope_head_dim", 0)
            if self.use_mla:
                return self.hf_text_config.kv_lora_rank + qk_rope_head_dim
            else:
                qk_nope_head_dim = getattr(self.hf_text_config, "qk_nope_head_dim", 0)
                if qk_rope_head_dim and qk_nope_head_dim:
                    return qk_rope_head_dim + qk_nope_head_dim

        if hasattr(self.hf_text_config, "model_type") and (
            self.hf_text_config.model_type == "zamba2"
        ):
            return self.hf_text_config.attention_head_dim

        if self.is_attention_free:
            return 0

        # NOTE: Some configs may set head_dim=None in the config
        if getattr(self.hf_text_config, "head_dim", None) is not None:
            return self.hf_text_config.head_dim

        # NOTE: Some models (such as PLaMo2.1) use `hidden_size_per_head`
        if getattr(self.hf_text_config, "hidden_size_per_head", None) is not None:
            return self.hf_text_config.hidden_size_per_head

        # FIXME(woosuk): This may not be true for all models.
        return (
            self.hf_text_config.hidden_size // self.hf_text_config.num_attention_heads
        )

    def get_total_num_kv_heads(self) -> int:
        """Returns the total number of KV heads."""
        # For GPTBigCode & Falcon:
        # NOTE: for falcon, when new_decoder_architecture is True, the
        # multi_query flag is ignored and we use n_head_kv for the number of
        # KV heads.
        falcon_model_types = ["falcon", "RefinedWeb", "RefinedWebModel"]
        new_decoder_arch_falcon = (
            self.hf_config.model_type in falcon_model_types
            and getattr(self.hf_config, "new_decoder_architecture", False)
        )
        if not new_decoder_arch_falcon and getattr(
            self.hf_text_config, "multi_query", False
        ):
            # Multi-query attention, only one KV head.
            # Currently, tensor parallelism is not supported in this case.
            return 1

        # For DBRX and MPT
        if self.hf_config.model_type == "mpt":
            if "kv_n_heads" in self.hf_config.attn_config:
                return self.hf_config.attn_config["kv_n_heads"]
            return self.hf_config.num_attention_heads
        if self.hf_config.model_type == "dbrx":
            return getattr(
                self.hf_config.attn_config,
                "kv_n_heads",
                self.hf_config.num_attention_heads,
            )

        if self.hf_config.model_type == "nemotron-nas":
            for block in self.hf_config.block_configs:
                if not block.attention.no_op:
                    return (
                        self.hf_config.num_attention_heads
                        // block.attention.n_heads_in_group
                    )

            raise RuntimeError("Couldn't determine number of kv heads")

        if self.is_attention_free:
            return 0

        attributes = [
            # For Falcon:
            "n_head_kv",
            "num_kv_heads",
            # For LLaMA-2:
            "num_key_value_heads",
            # For ChatGLM:
            "multi_query_group_num",
        ]
        for attr in attributes:
            num_kv_heads = getattr(self.hf_text_config, attr, None)
            if num_kv_heads is not None:
                return num_kv_heads

        # For non-grouped-query attention models, the number of KV heads is
        # equal to the number of attention heads.
        return self.hf_text_config.num_attention_heads

    def get_num_kv_heads(self, parallel_config: ParallelConfig) -> int:
        """Returns the number of KV heads per GPU."""
        if self.use_mla:
            # When using MLA during decode it becomes MQA
            return 1

        total_num_kv_heads = self.get_total_num_kv_heads()
        # If tensor parallelism is used, we divide the number of KV heads by
        # the tensor parallel size. We will replicate the KV heads in the
        # case where the number of KV heads is smaller than the tensor
        # parallel size so each GPU has at least one KV head.
        return max(1, total_num_kv_heads // parallel_config.tensor_parallel_size)

    def get_num_attention_heads(self, parallel_config: ParallelConfig) -> int:
        num_heads = getattr(self.hf_text_config, "num_attention_heads", 0)
        return num_heads // parallel_config.tensor_parallel_size

    def get_num_experts(self) -> int:
        """Returns the number of experts in the model."""
        num_expert_names = [
            "num_experts",  # Jamba
            "moe_num_experts",  # Dbrx
            "n_routed_experts",  # DeepSeek
            "num_local_experts",  # Mixtral
        ]
        num_experts = getattr_iter(self.hf_text_config, num_expert_names, 0)
        if isinstance(num_experts, list):
            # Ernie VL's remote code uses list[int]...
            # The values are always the same so we just take the first one.
            return num_experts[0]
        # Coerce to 0 if explicitly set to None
        return num_experts or 0

    def get_total_num_hidden_layers(self) -> int:
        if (
            self.hf_text_config.model_type == "deepseek_mtp"
            or self.hf_config.model_type == "mimo_mtp"
            or self.hf_config.model_type == "glm4_moe_mtp"
            or self.hf_config.model_type == "ernie_mtp"
            or self.hf_config.model_type == "qwen3_next_mtp"
            or self.hf_config.model_type == "pangu_ultra_moe_mtp"
        ):
            total_num_hidden_layers = getattr(
                self.hf_text_config, "num_nextn_predict_layers", 0
            )
        elif self.hf_config.model_type == "longcat_flash_mtp":
            total_num_hidden_layers = getattr(
                self.hf_text_config, "num_nextn_predict_layers", 1
            )
        else:
            total_num_hidden_layers = getattr(
                self.hf_text_config, "num_hidden_layers", 0
            )
        return total_num_hidden_layers

    def get_layers_start_end_indices(
        self, parallel_config: ParallelConfig
    ) -> tuple[int, int]:
        from vllm.distributed.utils import get_pp_indices

        total_num_hidden_layers = self.get_total_num_hidden_layers()

        # the layout order is: DP x PP x TP
        pp_rank = (
            parallel_config.rank // parallel_config.tensor_parallel_size
        ) % parallel_config.pipeline_parallel_size
        pp_size = parallel_config.pipeline_parallel_size
        start, end = get_pp_indices(total_num_hidden_layers, pp_rank, pp_size)
        return start, end

    def get_num_layers(self, parallel_config: ParallelConfig) -> int:
        start, end = self.get_layers_start_end_indices(parallel_config)
        return end - start

    def get_num_layers_by_block_type(
        self,
        parallel_config: ParallelConfig,
        block_type: LayerBlockType = "attention",
    ) -> int:
        # This function relies on 'layers_block_type' in hf_config,
        # for w/o this attribute, we will need to have workarounds like so
        attn_block_type = block_type == "attention"
        is_transformer = (
            not self.is_hybrid and not self.has_noops and not self.is_attention_free
        )
        start, end = self.get_layers_start_end_indices(parallel_config)

        if is_transformer:
            # Handle the basic case first
            return end - start if attn_block_type else 0
        elif self.is_attention_free:
            # Attention free
            # Note that this code assumes there
            # is only one type of attention-free block type.
            return 0 if attn_block_type else end - start
        elif self.has_noops:
            block_configs = self.hf_config.block_configs
            return sum(not bc.attention.no_op for bc in block_configs[start:end])
        else:
            # Hybrid model Jamba
            layers_block_type_value = getattr(
                self.hf_text_config, "layers_block_type", None
            )
            if layers_block_type_value is not None:
                if hasattr(self.hf_text_config, "model_type") and (
                    self.hf_text_config.model_type == "zamba2"
                ):
                    if attn_block_type:
                        return sum(
                            t == "hybrid" for t in layers_block_type_value[start:end]
                        )
                    else:
                        return self.get_num_layers(parallel_config)
                return sum(t == block_type for t in layers_block_type_value[start:end])

            # Hybrid model Minimax
            attn_type_list = getattr(self.hf_config, "attn_type_list", None)
            if attn_type_list:
                return sum(t == 1 for t in attn_type_list[start:end])

            # Hybrid model Qwen3Next
            layer_types_value = getattr(self.hf_config, "layer_types", None)
            if layer_types_value is not None:
                if block_type == "attention":
                    return sum(
                        t == "full_attention" for t in layer_types_value[start:end]
                    )
                elif block_type == "linear_attention":
                    return sum(
                        t == "linear_attention" for t in layer_types_value[start:end]
                    )
                else:
                    return sum(t == block_type for t in layer_types_value[start:end])

            if (
                layers_block_type_value is None
                and attn_type_list is None
                and layer_types_value is None
            ):
                raise ValueError(
                    "The model is an hybrid without a layers_block_type or an "
                    "attn_type_list, or a layer_types in the hf_config, "
                    f"cannot determine the num of {block_type} layers"
                )

    def get_mamba_chunk_size(self) -> int | None:
        """
        Returns the mamba chunk size if it exists
        """
        # used by e.g. Bamba, FalconH1, Granite, PLaMo2
        chunk_size = getattr(self.hf_text_config, "mamba_chunk_size", None)
        if chunk_size is None:
            # used by e.g. Mamba2, NemotronH, Zamba
            chunk_size = getattr(self.hf_text_config, "chunk_size", None)

        # Since Mamba1 does not have a chunk notion
        # we use a default chunk size of 1024.
        if chunk_size is None:
            chunk_size = 2048

        return chunk_size

    def get_multimodal_config(self) -> MultiModalConfig:
        """
        Get the multimodal configuration of the model.

        Raises:
            ValueError: If the model is not multimodal.
        """
        if self.multimodal_config is None:
            raise ValueError("The model is not multimodal.")

        return self.multimodal_config

    def try_get_generation_config(self) -> dict[str, Any]:
        """
        This method attempts to retrieve the non-default values of the
        generation config for this model.

        The generation config can contain information about special tokens, as
        well as sampling parameters. Which is why this method exists separately
        to `get_diff_sampling_param`.

        Returns:
            A dictionary containing the non-default generation config.
        """
        if self.generation_config in {"auto", "vllm"}:
            config = try_get_generation_config(
                self.hf_config_path or self.model,
                trust_remote_code=self.trust_remote_code,
                revision=self.revision,
                config_format=self.config_format,
            )
        else:
            config = try_get_generation_config(
                self.generation_config,
                trust_remote_code=self.trust_remote_code,
                config_format=self.config_format,
            )

        if config is None:
            return {}

        return config.to_diff_dict()

    def get_diff_sampling_param(self) -> dict[str, Any]:
        """
        This method returns a dictionary containing the non-default sampling
        parameters with `override_generation_config` applied.

        The default sampling parameters are:

        - vLLM's neutral defaults if `self.generation_config="vllm"`
        - the model's defaults if `self.generation_config="auto"`
        - as defined in `generation_config.json` if
            `self.generation_config="path/to/generation_config/dir"`

        Returns:
            A dictionary containing the non-default sampling parameters.
        """
        if self.generation_config == "vllm":
            config = {}
        else:
            config = self.try_get_generation_config()

        # Overriding with given generation config
        config.update(self.override_generation_config)

        available_params = [
            "repetition_penalty",
            "temperature",
            "top_k",
            "top_p",
            "min_p",
            "max_new_tokens",
        ]
        if any(p in config for p in available_params):
            diff_sampling_param = {
                p: config.get(p) for p in available_params if config.get(p) is not None
            }
            # Huggingface definition of max_new_tokens is equivalent
            # to vLLM's max_tokens
            if "max_new_tokens" in diff_sampling_param:
                diff_sampling_param["max_tokens"] = diff_sampling_param.pop(
                    "max_new_tokens"
                )
        else:
            diff_sampling_param = {}

        if diff_sampling_param:
            logger.warning_once(
                "Default sampling parameters have been overridden by the "
                "model's Hugging Face generation config recommended from the "
                "model creator. If this is not intended, please relaunch "
                "vLLM instance with `--generation-config vllm`."
            )
        return diff_sampling_param

    @property
    def is_encoder_decoder(self) -> bool:
        """Extract the HF encoder/decoder model flag."""
        return is_encoder_decoder(self.hf_config)

    @property
    def uses_alibi(self) -> bool:
        cfg = self.hf_text_config

        return (
            getattr(cfg, "alibi", False)  # Falcon
            or "BloomForCausalLM" in self.architectures  # Bloom
            or getattr(cfg, "position_encoding_type", "") == "alibi"  # codellm_1b_alibi
            or (
                hasattr(cfg, "attn_config")  # MPT
                and (
                    (
                        isinstance(cfg.attn_config, dict)
                        and cfg.attn_config.get("alibi", False)
                    )
                    or (
                        not isinstance(cfg.attn_config, dict)
                        and getattr(cfg.attn_config, "alibi", False)
                    )
                )
            )
        )

    @property
    def uses_mrope(self) -> bool:
        return uses_mrope(self.hf_config)

    @property
    def uses_xdrope_dim(self) -> int:
        return uses_xdrope_dim(self.hf_config)

    @property
    def is_multimodal_model(self) -> bool:
        return self.multimodal_config is not None

    @property
    def is_multimodal_raw_input_only_model(self) -> bool:
        return self._model_info.supports_multimodal_raw_input_only

    @property
    def is_cross_encoder(self) -> bool:
        return (
            self._model_info.supports_cross_encoding or self.convert_type == "classify"
        )

    @property
    def is_pp_supported(self) -> bool:
        return self._model_info.supports_pp

    @property
    def is_attention_free(self) -> bool:
        return self._model_info.is_attention_free

    @property
    def is_hybrid(self) -> bool:
        # Handle granite-4.0-micro case which uses hybrid config but does not
        # actually contain any non-attention layers.
        layer_types = getattr(self.hf_config, "layer_types", None)
        if layer_types is not None and all(
            layer == "attention" for layer in layer_types
        ):
            return False
        return self._model_info.is_hybrid

    @property
    def has_noops(self) -> bool:
        return self._model_info.has_noops

    @property
    def has_inner_state(self):
        return self._model_info.has_inner_state

    @property
    def supports_mamba_prefix_caching(self) -> bool:
        return self._model_info.supports_mamba_prefix_caching

    @property
    def use_mla(self) -> bool:
        return self.is_deepseek_mla and not envs.VLLM_MLA_DISABLE

    @property
    def is_matryoshka(self) -> bool:
        return bool(getattr(self.hf_config, "matryoshka_dimensions", None)) or getattr(
            self.hf_config, "is_matryoshka", False
        )

    @property
    def matryoshka_dimensions(self):
        return getattr(self.hf_config, "matryoshka_dimensions", None)

    @property
    def use_pad_token(self) -> bool:
        # cross_encoder models defaults to using pad_token.
        # `llm as reranker` models defaults to not using pad_token.
        return getattr(self.hf_config, "use_pad_token", True)

    @property
    def head_dtype(self) -> torch.dtype:
        """
        "head" refers to the last Linear layer(s) of an LLM,
        such as the lm_head in a generation model,
        or the score or classifier in a classification model.

        `head_dtype` currently only supports pooling models.\n
        - The pooling model defaults to using fp32 head,
        you can use --hf-overrides '{"head_dtype": "model"}' to disable it.
        """

        head_dtype = _get_head_dtype(
            config=self.hf_config, dtype=self.dtype, runner_type=self.runner_type
        )

        if self.runner_type != "pooling" and head_dtype != self.dtype:
            logger.warning_once(
                "`head_dtype` currently only supports pooling models."
                "fallback to model dtype [%s].",
                self.dtype,
            )
            return self.dtype

        if head_dtype not in current_platform.supported_dtypes:
            logger.warning_once(
                "The current platform does not support [%s] head dtype, "
                "fallback to model dtype [%s].",
                head_dtype,
                self.dtype,
            )
            return self.dtype

        logger.debug_once("head dtype: %s", head_dtype)
        return head_dtype

    @property
    def hidden_size(self):
        if hasattr(self.hf_config, "hidden_size"):
            return self.hf_config.hidden_size
        text_config = self.hf_config.get_text_config()
        return text_config.hidden_size

    @property
    def embedding_size(self):
        dense_modules = try_get_dense_modules(self.model, revision=self.revision)
        if dense_modules is not None:
            return dense_modules[-1]["out_features"]
        return self.hidden_size

    def get_and_verify_max_len(self, max_model_len: int):
        # Consider max_model_len in tokenizer_config only when
        # pooling models use absolute position_embedding.
        tokenizer_config = None
        if (
            self.runner_type == "pooling"
            and getattr(self.hf_config, "position_embedding_type", "") == "absolute"
        ):
            tokenizer_config = try_get_tokenizer_config(
                self.tokenizer,
                trust_remote_code=self.trust_remote_code,
                revision=self.tokenizer_revision,
            )
        max_model_len = _get_and_verify_max_len(
            hf_config=self.hf_text_config,
            tokenizer_config=tokenizer_config,
            max_model_len=max_model_len,
            disable_sliding_window=self.disable_sliding_window,
            sliding_window=self.get_sliding_window(),
            spec_target_max_model_len=self.spec_target_max_model_len,
            encoder_config=self.encoder_config,
        )
        logger.info("Using max model len %s", max_model_len)
        return max_model_len


def get_served_model_name(model: str, served_model_name: str | list[str] | None):
    """
    If the input is a non-empty list, the first model_name in
    `served_model_name` is taken.
    If the input is a non-empty string, it is used directly.
    For cases where the input is either an empty string or an
    empty list, the fallback is to use `self.model`.
    """
    if not served_model_name:
        return model
    if isinstance(served_model_name, list):
        return served_model_name[0]
    return served_model_name


# Some model suffixes are based on auto classes from Transformers:
# https://huggingface.co/docs/transformers/en/model_doc/auto
# NOTE: Items higher on this list priority over lower ones
_SUFFIX_TO_DEFAULTS: list[tuple[str, tuple[RunnerType, ConvertType]]] = [
    ("ForCausalLM", ("generate", "none")),
    ("ForConditionalGeneration", ("generate", "none")),
    ("ChatModel", ("generate", "none")),
    ("LMHeadModel", ("generate", "none")),
    ("ForTextEncoding", ("pooling", "embed")),
    ("EmbeddingModel", ("pooling", "embed")),
    ("ForSequenceClassification", ("pooling", "classify")),
    ("ForAudioClassification", ("pooling", "classify")),
    ("ForImageClassification", ("pooling", "classify")),
    ("ForVideoClassification", ("pooling", "classify")),
    ("ClassificationModel", ("pooling", "classify")),
    ("ForRewardModeling", ("pooling", "reward")),
    ("RewardModel", ("pooling", "reward")),
    # Let other `*Model`s take priority
    ("Model", ("pooling", "embed")),
]


def iter_architecture_defaults():
    yield from _SUFFIX_TO_DEFAULTS


def try_match_architecture_defaults(
    architecture: str,
    *,
    runner_type: RunnerType | None = None,
    convert_type: ConvertType | None = None,
) -> tuple[str, tuple[RunnerType, ConvertType]] | None:
    for suffix, (
        default_runner_type,
        default_convert_type,
    ) in iter_architecture_defaults():
        if (
            (runner_type is None or runner_type == default_runner_type)
            and (convert_type is None or convert_type == default_convert_type)
            and architecture.endswith(suffix)
        ):
            return suffix, (default_runner_type, default_convert_type)

    return None


_STR_DTYPE_TO_TORCH_DTYPE = {
    "half": torch.float16,
    "float16": torch.float16,
    "float": torch.float32,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}

# model_type -> reason
_FLOAT16_NOT_SUPPORTED_MODELS = {
    "gemma2": "Numerical instability. Please use bfloat16 or float32 instead.",
    "gemma3": "Numerical instability. Please use bfloat16 or float32 instead.",
    "gemma3_text": "Numerical instability. Please use bfloat16 or float32 instead.",
    "plamo2": "Numerical instability. Please use bfloat16 or float32 instead.",
    "glm4": "Numerical instability. Please use bfloat16 or float32 instead.",
}


def _is_valid_dtype(model_type: str, dtype: torch.dtype):
    if model_type in _FLOAT16_NOT_SUPPORTED_MODELS and dtype == torch.float16:  # noqa: E501, SIM103
        return False

    return True


def _check_valid_dtype(model_type: str, dtype: torch.dtype):
    if model_type in _FLOAT16_NOT_SUPPORTED_MODELS and dtype == torch.float16:
        reason = _FLOAT16_NOT_SUPPORTED_MODELS[model_type]
        raise ValueError(
            f"The model type {model_type!r} does not support float16. Reason: {reason}"
        )

    return True


def _find_dtype(
    model_id: str,
    config: PretrainedConfig,
    *,
    revision: str | None,
):
    # NOTE: getattr(config, "dtype", torch.float32) is not correct
    # because config.dtype can be None.
    config_dtype = getattr(config, "dtype", None)

    # Fallbacks for multi-modal models if the root config
    # does not define dtype
    if config_dtype is None:
        config_dtype = getattr(config.get_text_config(), "dtype", None)
    if config_dtype is None and hasattr(config, "vision_config"):
        config_dtype = getattr(config.vision_config, "dtype", None)
    if config_dtype is None and hasattr(config, "encoder_config"):
        config_dtype = getattr(config.encoder_config, "dtype", None)

    # Try to read the dtype of the weights if they are in safetensors format
    if config_dtype is None:
        repo_mt = try_get_safetensors_metadata(model_id, revision=revision)

        if repo_mt and (files_mt := repo_mt.files_metadata):
            param_dtypes: set[torch.dtype] = {
                _SAFETENSORS_TO_TORCH_DTYPE[dtype_str]
                for file_mt in files_mt.values()
                for dtype_str in file_mt.parameter_count
                if dtype_str in _SAFETENSORS_TO_TORCH_DTYPE
            }

            if param_dtypes:
                return common_broadcastable_dtype(param_dtypes)

    if config_dtype is None:
        config_dtype = torch.float32

    return config_dtype


def _resolve_auto_dtype(
    model_type: str,
    config_dtype: torch.dtype,
    *,
    is_pooling_model: bool,
):
    from vllm.platforms import current_platform

    supported_dtypes = [
        dtype
        for dtype in current_platform.supported_dtypes
        if _is_valid_dtype(model_type, dtype)
    ]

    if is_pooling_model and torch.float16 in supported_dtypes:
        preferred_dtype = torch.float16
    else:
        preferred_dtype = supported_dtypes[0]

    # Downcast for float32 models
    if config_dtype == torch.float32:
        config_dtype = preferred_dtype

    if config_dtype in supported_dtypes:
        return config_dtype

    # Ensure device compatibility
    device_name = current_platform.get_device_name()
    device_capability = current_platform.get_device_capability()

    if device_capability is None:
        device_str = f"{device_name!r}"
    else:
        version_str = device_capability.as_version_str()
        device_str = f"{device_name!r} (with compute capability {version_str})"

    logger.warning(
        "Your device %s doesn't support %s. Falling back to %s for compatibility.",
        device_str,
        config_dtype,
        preferred_dtype,
    )

    return preferred_dtype


def _get_and_verify_dtype(
    model_id: str,
    config: PretrainedConfig,
    dtype: str | torch.dtype,
    *,
    is_pooling_model: bool,
    revision: str | None = None,
) -> torch.dtype:
    config_dtype = _find_dtype(model_id, config, revision=revision)
    model_type = config.model_type

    if isinstance(dtype, str):
        dtype = dtype.lower()
        if dtype == "auto":
            # Set default dtype from model config
            torch_dtype = _resolve_auto_dtype(
                model_type,
                config_dtype,
                is_pooling_model=is_pooling_model,
            )
        else:
            if dtype not in _STR_DTYPE_TO_TORCH_DTYPE:
                raise ValueError(f"Unknown dtype: {dtype!r}")
            torch_dtype = _STR_DTYPE_TO_TORCH_DTYPE[dtype]
    elif isinstance(dtype, torch.dtype):
        torch_dtype = dtype
    else:
        raise ValueError(f"Unknown dtype: {dtype}")

    _check_valid_dtype(model_type, torch_dtype)

    if torch_dtype != config_dtype:
        if torch_dtype == torch.float32:
            # Upcasting to float32 is allowed.
            logger.info("Upcasting %s to %s.", config_dtype, torch_dtype)
        elif config_dtype == torch.float32:
            # Downcasting from float32 to float16 or bfloat16 is allowed.
            logger.info("Downcasting %s to %s.", config_dtype, torch_dtype)
        else:
            # Casting between float16 and bfloat16 is allowed with a warning.
            logger.warning("Casting %s to %s.", config_dtype, torch_dtype)

    return torch_dtype


def _get_head_dtype(
    config: PretrainedConfig, dtype: torch.dtype, runner_type: str
) -> torch.dtype:
    head_dtype: str | torch.dtype | None = getattr(config, "head_dtype", None)

    if head_dtype == "model":
        return dtype
    elif isinstance(head_dtype, str):
        head_dtype = head_dtype.lower()
        if head_dtype not in _STR_DTYPE_TO_TORCH_DTYPE:
            raise ValueError(f"Unknown dtype: {head_dtype!r}")
        return _STR_DTYPE_TO_TORCH_DTYPE[head_dtype]
    elif isinstance(head_dtype, torch.dtype):
        return head_dtype
    elif head_dtype is None:
        if torch.float32 not in current_platform.supported_dtypes:
            return dtype
        if runner_type == "pooling":
            return torch.float32
        return dtype
    else:
        raise ValueError(f"Unknown dtype: {head_dtype}")


def _get_and_verify_max_len(
    hf_config: PretrainedConfig,
    tokenizer_config: dict | None,
    max_model_len: int | None,
    disable_sliding_window: bool,
    sliding_window: int | None,
    spec_target_max_model_len: int | None = None,
    encoder_config: Any | None = None,
) -> int:
    """Get and verify the model's maximum length."""
    derived_max_model_len = float("inf")
    possible_keys = [
        # OPT
        "max_position_embeddings",
        # GPT-2
        "n_positions",
        # MPT
        "max_seq_len",
        # ChatGLM2
        "seq_length",
        # Command-R
        "model_max_length",
        # Whisper
        "max_target_positions",
        # Others
        "max_sequence_length",
        "max_seq_length",
        "seq_len",
    ]
    # Choose the smallest "max_length" from the possible keys
    max_len_key = None
    for key in possible_keys:
        max_len = getattr(hf_config, key, None)
        if max_len is not None:
            max_len_key = key if max_len < derived_max_model_len else max_len_key
            derived_max_model_len = min(derived_max_model_len, max_len)
    # For Command-R / Cohere, Cohere2 / Aya Vision models
    if tmp_max_len := getattr(hf_config, "model_max_length", None):
        max_len_key = "model_max_length"
        derived_max_model_len = tmp_max_len

    # If sliding window is manually disabled, max_length should be less
    # than the sliding window length in the model config.
    if (
        disable_sliding_window
        and sliding_window is not None
        and sliding_window < derived_max_model_len
    ):
        max_len_key = "sliding_window"
        derived_max_model_len = sliding_window

    # Consider model_max_length in tokenizer_config
    if tokenizer_config:
        tokenizer_model_max_length = tokenizer_config.get(
            "model_max_length", derived_max_model_len
        )
        derived_max_model_len = min(derived_max_model_len, tokenizer_model_max_length)

    # If none of the keys were found in the config, use a default and
    # log a warning.
    if derived_max_model_len == float("inf"):
        if max_model_len is not None:
            # If max_model_len is specified, we use it.
            return max_model_len

        if spec_target_max_model_len is not None:
            # If this is a speculative draft model, we use the max model len
            # from the target model.
            return spec_target_max_model_len

        default_max_len = 2048
        logger.warning(
            "The model's config.json does not contain any of the following "
            "keys to determine the original maximum length of the model: "
            "%s. Assuming the model's maximum length is %d.",
            possible_keys,
            default_max_len,
        )
        derived_max_model_len = default_max_len

    # In Transformers v5 rope_parameters could be TypedDict or dict[str, TypedDict].
    # To simplify the verification, we convert it to dict[str, TypedDict].
    rope_parameters = getattr(hf_config, "rope_parameters", None)
    if rope_parameters and not set(rope_parameters.keys()).issubset(
        ALLOWED_LAYER_TYPES
    ):
        rope_parameters = {"": rope_parameters}

    # NOTE(woosuk): Gemma3's max_model_len (128K) is already scaled by RoPE
    # scaling, so we skip applying the scaling factor again.
    if rope_parameters is not None and "gemma3" not in hf_config.model_type:
        scaling_factor = 1.0
        for rp in rope_parameters.values():
            # No need to consider "type" key because of patch_rope_parameters when
            # loading HF config
            rope_type = rp["rope_type"]

            if rope_type not in ("su", "longrope", "llama3"):
                # NOTE: rope_type == "default" does not define factor https://github.com/huggingface/transformers/blob/v4.45.2/src/transformers/modeling_rope_utils.py
                # NOTE: This assumes all layer types have the same scaling factor.
                scaling_factor = rp.get("factor", scaling_factor)

                if rope_type == "yarn":
                    derived_max_model_len = rp["original_max_position_embeddings"]
        # Do this outside loop since all layer types should have the same scaling
        derived_max_model_len *= scaling_factor

    if encoder_config and "max_seq_length" in encoder_config:
        derived_max_model_len = encoder_config["max_seq_length"]

    # If the user didn't specify `max_model_len`, then use that derived from
    # the model config as a default value.
    if max_model_len is None:
        # For LongRoPE, default to original_max_position_embeddings to avoid
        # performance degradation for shorter sequences
        if rope_parameters is not None and any(
            rp["rope_type"] == "longrope" for rp in rope_parameters.values()
        ):
            max_model_len = int(
                getattr(
                    hf_config, "original_max_position_embeddings", derived_max_model_len
                )
            )
        else:
            max_model_len = int(derived_max_model_len)
        max_model_len = current_platform.check_max_model_len(max_model_len)

    # If the user specified a max length, make sure it is smaller than the
    # derived length from the HF model config.
    elif max_model_len > derived_max_model_len:
        # Some models might have a separate key for specifying model_max_length
        # that will be bigger than derived_max_model_len. We compare user input
        # with model_max_length and allow this override when it's smaller.
        model_max_length = getattr(hf_config, "model_max_length", None)
        if model_max_length is None or max_model_len > model_max_length:
            msg = (
                f"User-specified max_model_len ({max_model_len}) is greater "
                f"than the derived max_model_len ({max_len_key}="
                f"{derived_max_model_len} or model_max_length="
                f"{model_max_length} in model's config.json)."
            )
            warning = (
                "VLLM_ALLOW_LONG_MAX_MODEL_LEN must be used with extreme "
                "caution. If the model uses relative position encoding (RoPE), "
                "positions exceeding derived_max_model_len lead to nan. If the "
                "model uses absolute position encoding, positions exceeding "
                "derived_max_model_len will cause a CUDA array out-of-bounds "
                "error."
            )
            if envs.VLLM_ALLOW_LONG_MAX_MODEL_LEN:
                logger.warning_once("%s %s", msg, warning)
            else:
                raise ValueError(
                    f"{msg} To allow overriding this maximum, set "
                    f"the env var VLLM_ALLOW_LONG_MAX_MODEL_LEN=1. {warning}"
                )
    return int(max_model_len)
