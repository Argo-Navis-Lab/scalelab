"""The Track S tokenizer: a tiny fixed vocabulary in which almost every token is one character."""

import re
from collections.abc import Iterable

# Token ids are positions in VOCAB. Never reorder or remove entries: every saved dataset and
# every trained model would silently mean something else. A new token may only be added at the end.
SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>")
CHARACTERS = "0123456789+-*/()=,;:#"
VOCAB = SPECIAL_TOKENS + tuple(CHARACTERS)

# Rows in every model's embedding table: more than VOCAB uses, so a token can be added later
# without changing the shape of any model.
VOCAB_SIZE = 64

PAD = VOCAB.index("<pad>")  # fills the gaps when sequences of different lengths share a batch
BOS = VOCAB.index("<bos>")  # "beginning of sequence": the first token of every prompt
EOS = VOCAB.index("<eos>")  # "end of sequence": the model writes it when its answer is finished

_IDS = {token: i for i, token in enumerate(VOCAB)}
_PIECE = re.compile(r"<[a-z]+>|.", re.DOTALL)  # a special token such as <bos>, or one character


def encode(text: str) -> list[int]:
    """Turn text into token ids. Special tokens are written by name, as in "<bos>12+7=".

    Anything outside the vocabulary (a letter, a space, a decimal point) is an error, never
    silently skipped: a mistake in a task's text format should fail loudly.
    """
    ids: list[int] = []
    for match in _PIECE.finditer(text):
        piece = match.group()
        if piece not in _IDS:
            raise ValueError(f"can't encode {piece!r} (position {match.start()} of {text!r})")
        ids.append(_IDS[piece])
    return ids


def decode(ids: Iterable[int]) -> str:
    """Turn token ids back into text, so that decode(encode(text)) == text."""
    pieces: list[str] = []
    for i in ids:
        if not 0 <= i < len(VOCAB):  # plain VOCAB[-1] would quietly give the last token
            raise ValueError(f"no token has id {i} (ids run from 0 to {len(VOCAB) - 1})")
        pieces.append(VOCAB[i])
    return "".join(pieces)
