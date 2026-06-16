# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the FP32 LM-head path in LogitsProcessor (--enable-fp32-lm-head).

Probes the matmul that produces logits and asserts both operands are FP32 when
the flag is on, and stay in the activation dtype when off. CPU-friendly: builds
the layer without distributed init and uses an identity TP gather.
"""

from types import SimpleNamespace
from typing import Any

import pytest
import torch

import vllm.model_executor.layers.logits_processor as lp_mod
from vllm.model_executor.layers.logits_processor import LogitsProcessor


class _LMHeadStub:
    """Plain unquantized head: just a float weight matrix."""

    def __init__(self, vocab, hidden, dtype):
        self.weight = torch.randn(vocab, hidden, dtype=dtype)
        self.quant_method: Any = None


def _make_processor(use_fp32, vocab_size):
    """Build a LogitsProcessor without running __init__ (avoids platform/config
    machinery), setting only the attributes _get_logits relies on."""
    proc = object.__new__(LogitsProcessor)
    proc.use_fp32_lm_head = use_fp32
    proc.org_vocab_size = vocab_size
    proc.scale = 1.0
    proc.soft_cap = None
    # No TP in the test: gather is identity.
    proc._gather_logits = lambda logits: logits
    return proc


def _probe_get_logits(proc, hidden_states, lm_head, monkeypatch):
    captured: dict[str, torch.dtype] = {}
    original_matmul = torch.matmul

    def probe_matmul(a, b, *args, **kwargs):
        captured.setdefault("a", a.dtype)
        captured.setdefault("b", b.dtype)
        return original_matmul(a, b, *args, **kwargs)

    monkeypatch.setattr(torch, "matmul", probe_matmul)
    logits = proc._get_logits(hidden_states, lm_head, None)
    return logits, captured


@pytest.mark.parametrize("activation_dtype", [torch.float16, torch.bfloat16])
def test_fp32_lm_head_upcasts(activation_dtype, monkeypatch):
    batch, hidden, vocab = 2, 64, 128
    hidden_states = torch.randn(batch, hidden, dtype=activation_dtype)
    lm_head = _LMHeadStub(vocab, hidden, activation_dtype)
    proc = _make_processor(use_fp32=True, vocab_size=vocab)

    logits, captured = _probe_get_logits(proc, hidden_states, lm_head, monkeypatch)

    assert captured["a"] == torch.float32
    assert captured["b"] == torch.float32
    assert logits.dtype == torch.float32
    assert logits.shape == (batch, vocab)


@pytest.mark.parametrize("activation_dtype", [torch.float16, torch.bfloat16])
def test_fp32_lm_head_disabled_keeps_dtype(activation_dtype, monkeypatch):
    # When disabled, _get_logits delegates to lm_head.quant_method.apply, which
    # computes in the activation dtype.
    class _QuantMethod:
        def apply(self, layer, x, bias=None):
            return torch.matmul(x.to(layer.weight.dtype), layer.weight.t())

    batch, hidden, vocab = 2, 64, 128
    hidden_states = torch.randn(batch, hidden, dtype=activation_dtype)
    lm_head = _LMHeadStub(vocab, hidden, activation_dtype)
    lm_head.quant_method = _QuantMethod()
    proc = _make_processor(use_fp32=False, vocab_size=vocab)

    logits, captured = _probe_get_logits(proc, hidden_states, lm_head, monkeypatch)

    assert captured["a"] == activation_dtype
    assert captured["b"] == activation_dtype
    assert logits.dtype == activation_dtype
    assert logits.shape == (batch, vocab)


def test_fp32_lm_head_falls_back_for_quantized_weight(monkeypatch):
    # A non-float (e.g. int8) weight must not take the fp32 matmul path; it
    # should fall back to quant_method.apply.
    batch, hidden, vocab = 2, 64, 128
    hidden_states = torch.randn(batch, hidden, dtype=torch.bfloat16)
    lm_head = SimpleNamespace(weight=torch.zeros(vocab, hidden, dtype=torch.int8))

    apply_called = {"v": False}

    class _QuantMethod:
        def apply(self, layer, x, bias=None):
            apply_called["v"] = True
            return torch.zeros(x.shape[0], layer.weight.shape[0], dtype=x.dtype)

    lm_head.quant_method = _QuantMethod()
    proc = _make_processor(use_fp32=True, vocab_size=vocab)

    logits = proc._get_logits(hidden_states, lm_head, None)

    assert apply_called["v"] is True
    assert logits.shape == (batch, vocab)


def test_init_reads_flag_from_vllm_config(monkeypatch):
    # __init__ should pick up enable_fp32_lm_head from the ambient VllmConfig.
    fake_cfg = SimpleNamespace(model_config=SimpleNamespace(enable_fp32_lm_head=True))
    monkeypatch.setattr(lp_mod, "get_current_vllm_config", lambda: fake_cfg)

    proc = LogitsProcessor(vocab_size=128)
    assert proc.use_fp32_lm_head is True

    fake_cfg.model_config.enable_fp32_lm_head = False
    proc = LogitsProcessor(vocab_size=128)
    assert proc.use_fp32_lm_head is False
