"""Rollouts: several sampled answers to each prompt, written with the KV cache and kept with
everything RL needs to score them and learn from them."""

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from scalelab.core.config import SamplingParams
from scalelab.models.transformer import Transformer
from scalelab.rl.sampling import sample
from scalelab.tasks.tokenizer import EOS, PAD

FinishReason = Literal["eos", "length"]  # wrote <eos>, or ran out of context first


@dataclass(frozen=True)
class RolloutBatch:
    """n sampled answers to each of several prompts, one row per answer.

    Each row holds its prompt, then the answer the model wrote, then PAD up to the longest
    row. Rows come in groups: the n answers to the first prompt, then the n to the second,
    and so on. All tensors are on the CPU.
    """

    tokens: Tensor  # (rows, width) prompt, answer (with <eos> if written), then PAD
    prompt_lengths: Tensor  # (rows,)
    lengths: Tensor  # (rows,) prompt and answer together
    finish_reasons: list[FinishReason]
    sampler_logprobs: Tensor  # (rows, width) each answer token's log-prob as drawn; 0 elsewhere
    group_ids: Tensor  # (rows,) the index of the prompt each row answers
    policy_version: int  # which weights wrote the answers

    @property
    def response_mask(self) -> Tensor:
        """(rows, width) True at the answer's tokens: the ones RL learns from."""
        positions = torch.arange(self.tokens.shape[1])
        return (positions >= self.prompt_lengths[:, None]) & (positions < self.lengths[:, None])

    def rows(self, start: int, stop: int) -> "RolloutBatch":
        """Rows start to stop (stop not included) on their own, cut to the width their
        longest row needs: a micro-batch for training."""
        width = int(self.lengths[start:stop].max())
        return RolloutBatch(
            tokens=self.tokens[start:stop, :width],
            prompt_lengths=self.prompt_lengths[start:stop],
            lengths=self.lengths[start:stop],
            finish_reasons=self.finish_reasons[start:stop],
            sampler_logprobs=self.sampler_logprobs[start:stop, :width],
            group_ids=self.group_ids[start:stop],
            policy_version=self.policy_version,
        )

    def answers(self) -> list[list[int]]:
        """Each row's answer tokens, with <eos> at the end when it was written."""
        starts, ends = self.prompt_lengths.tolist(), self.lengths.tolist()
        rows = zip(self.tokens.tolist(), starts, ends, strict=True)
        return [row[start:end] for row, start, end in rows]


@torch.no_grad()
def generate(
    model: Transformer,
    prompts: Sequence[Sequence[int]],
    n_per_prompt: int,
    params: SamplingParams,
    generator: torch.Generator,
    policy_version: int = 0,
    batch_size: int = 512,
) -> RolloutBatch:
    """Sample `n_per_prompt` answers to every prompt. Each answer ends at its first <eos>, or
    where prompt and answer fill the model's context.

    Rows whose prompts have the same length are decoded together, at most `batch_size` at a
    time, so no padding is needed. All the randomness comes from `generator` (on the model's
    device): the same prompts, settings and generator state give the same rollouts.
    """
    # groups[r]: the prompt that row r answers. n rows per prompt, in the prompts' order.
    groups = [index for index in range(len(prompts)) for _ in range(n_per_prompt)]
    same_length: dict[int, list[int]] = defaultdict(list)  # prompt length -> those rows
    for row, index in enumerate(groups):
        same_length[len(prompts[index])].append(row)
    answers: list[list[int]] = [[] for _ in groups]
    logprobs: list[list[float]] = [[] for _ in groups]
    for rows in same_length.values():
        for first in range(0, len(rows), batch_size):
            chunk = rows[first : first + batch_size]
            chunk_prompts = [prompts[groups[row]] for row in chunk]
            written = _sample_batch(model, chunk_prompts, params, generator)
            for row, (answer, values) in zip(chunk, written, strict=True):
                answers[row], logprobs[row] = answer, values  # back to the row's own place

    prompt_lengths = [len(prompts[index]) for index in groups]
    lengths = [start + len(answer) for start, answer in zip(prompt_lengths, answers, strict=True)]
    tokens = torch.full((len(groups), max(lengths)), PAD)
    sampler_logprobs = torch.zeros(len(groups), max(lengths))
    for row, index in enumerate(groups):
        start, end = prompt_lengths[row], lengths[row]
        tokens[row, :end] = torch.tensor([*prompts[index], *answers[row]])
        sampler_logprobs[row, start:end] = torch.tensor(logprobs[row])
    return RolloutBatch(
        tokens=tokens,
        prompt_lengths=torch.tensor(prompt_lengths),
        lengths=torch.tensor(lengths),
        finish_reasons=["eos" if answer[-1] == EOS else "length" for answer in answers],
        sampler_logprobs=sampler_logprobs,
        group_ids=torch.tensor(groups),
        policy_version=policy_version,
    )


def _sample_batch(
    model: Transformer,
    prompts: list[Sequence[int]],
    params: SamplingParams,
    generator: torch.Generator,
) -> list[tuple[list[int], list[float]]]:
    """For prompts of the same length: each one's sampled answer, and the log-prob of each
    answer token as it was drawn."""
    device = model.embed.weight.device
    cache = model.new_cache(len(prompts))
    logits = model.decode(torch.tensor(prompts, device=device), cache)[:, -1]  # read the prompts
    written: list[Tensor] = []
    logp: list[Tensor] = []
    finished = torch.zeros(len(prompts), dtype=torch.bool, device=device)
    while True:
        tokens, token_logp = sample(logits, params, generator)
        written.append(tokens)
        logp.append(token_logp)
        finished |= tokens == EOS
        if finished.all() or cache.length + 1 == cache.max_len:  # all done, or context full
            break
        logits = model.decode(tokens[:, None], cache)[:, -1]
    rows: list[list[int]] = torch.stack(written, dim=1).tolist()
    values: list[list[float]] = torch.stack(logp, dim=1).tolist()
    results: list[tuple[list[int], list[float]]] = []
    for row, row_values in zip(rows, values, strict=True):
        end = row.index(EOS) + 1 if EOS in row else len(row)  # what follows <eos> is dropped
        results.append((row[:end], row_values[:end]))
    return results
