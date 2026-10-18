"""Starting weights for the Track S transformer."""

import math

from torch import nn

from scalelab.models.norms import RMSNorm

STD = 0.02  # the spread of the starting weights, as in GPT-2


def init_weights(model: nn.Module, n_layers: int) -> None:
    """GPT-2's recipe: every Linear and Embedding weight drawn from a normal distribution with
    standard deviation 0.02, and every RMSNorm weight set to 1.

    The two layers that add into the running sum of each block ("attn.o" and "mlp.down") start
    smaller, by 1 / sqrt(2 * n_layers): a model with 2 * n_layers such additions then starts
    with a running sum about as large as a shallow model's. It draws from PyTorch's global
    random generator, which seed_everything() seeds from the run's "init" stream.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear | nn.Embedding):
            residual = name.endswith(("attn.o", "mlp.down"))
            std = STD / math.sqrt(2 * n_layers) if residual else STD
            nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)
