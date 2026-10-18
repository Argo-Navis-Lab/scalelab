"""Tokenizer tests."""

import pytest

from scalelab.tasks.tokenizer import BOS, EOS, PAD, VOCAB, VOCAB_SIZE, decode, encode

# --- the vocabulary ------------------------------------------------------------------------


def test_vocab_is_pinned():
    # Golden value: a token's id is its position here. If this ever changes, every saved
    # dataset and every trained model silently means something else. Only ever append.
    assert VOCAB == ("<pad>", "<bos>", "<eos>", *"0123456789+-*/()=,;:#")
    assert (PAD, BOS, EOS) == (0, 1, 2)


def test_vocab_fits_the_embedding_table():
    assert VOCAB_SIZE == 64
    assert len(VOCAB) <= VOCAB_SIZE
    assert len(set(VOCAB)) == len(VOCAB)  # no token appears twice


# --- encoding ------------------------------------------------------------------------------


def test_digits_and_symbols():
    assert encode("0123") == [3, 4, 5, 6]
    assert encode("12+7=19") == [4, 5, 13, 10, 19, 4, 12]


def test_special_tokens_are_one_id_each():
    assert encode("<bos>42<eos>") == [BOS, 7, 5, EOS]


def test_every_token_encodes_to_its_own_id():
    for i, token in enumerate(VOCAB):
        assert encode(token) == [i]


@pytest.mark.parametrize("text", ["", "7", "<bos>347+85=", "(25-3)*7=154;#154<eos>", "<pad><pad>"])
def test_decode_undoes_encode(text: str):
    assert decode(encode(text)) == text


# --- mistakes are caught -------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    ["x", "1 2", "1\n2", "3.5", "<cls>", "<bos", "π"],
    ids=["letter", "space", "newline", "decimal", "unknown-special", "unclosed-special", "unicode"],
)
def test_text_outside_the_vocab_is_refused(text: str):
    with pytest.raises(ValueError, match="can't encode"):
        encode(text)


def test_the_error_says_where():
    with pytest.raises(ValueError, match=r"'x' \(position 2 of '12x'\)"):
        encode("12x")


@pytest.mark.parametrize(
    "bad_id", [-1, len(VOCAB), VOCAB_SIZE], ids=["negative", "first-unused", "past-the-table"]
)
def test_decode_refuses_ids_without_a_token(bad_id: int):
    with pytest.raises(ValueError, match="no token has id"):
        decode([bad_id])
