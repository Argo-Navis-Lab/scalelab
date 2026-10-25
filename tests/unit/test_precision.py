"""The number formats models run in (m3s19): float32, or float16 under autocast."""

import pytest
import torch
import torch.nn.functional as F

from scalelab.core.config import ModelConfig
from scalelab.costmodel.specs import Precision
from scalelab.models.precision import DTYPES, autocast
from scalelab.models.transformer import Transformer

CPU = torch.device("cpu")
TINY = ModelConfig(name="tiny", n_layers=2, d_model=64, n_heads=1, context=32)


def test_fp32_changes_nothing() -> None:
    with autocast("fp32", CPU):
        assert not torch.is_autocast_enabled("cpu")


def test_fp16_is_autocasts_float16_for_the_region() -> None:
    x, weight = torch.randn(2, 3), torch.randn(4, 3)
    with autocast("fp16", CPU):
        assert torch.get_autocast_dtype("cpu") == torch.float16
        assert F.linear(x, weight).dtype == torch.float16
    assert not torch.is_autocast_enabled("cpu")
    assert F.linear(x, weight).dtype == torch.float32


@pytest.mark.parametrize("precision", ["fp32", "fp16"])
def test_a_cache_is_made_in_the_dtype_attention_computes_in(precision: Precision) -> None:
    with torch.random.fork_rng():
        model = Transformer(TINY)
    with autocast(precision, CPU):
        cache = model.new_cache(2)
    assert {layer.k.dtype for layer in cache.layers} == {DTYPES[precision]}
    assert {layer.v.dtype for layer in cache.layers} == {DTYPES[precision]}
    assert model.new_cache(2).layers[0].k.dtype == torch.float32  # outside: the model's own
