import pytest

from scratch_transformer.tokenizer import BPETokenizer


def byte_vocab() -> dict[int, bytes]:
    return {i: bytes([i]) for i in range(256)}


def test_decode_byte_ids_returns_utf8_text() -> None:
    tokenizer = BPETokenizer(vocab=byte_vocab(), merges={})

    assert tokenizer.decode([65, 10]) == "A\n"
    assert tokenizer.decode(list("é".encode("utf-8"))) == "é"


def test_decode_merged_token_ids_returns_text() -> None:
    vocab = byte_vocab()
    vocab[256] = b"ab"
    vocab[257] = b"abc"
    tokenizer = BPETokenizer(
        vocab=vocab,
        merges={
            (ord("a"), ord("b")): 256,
            (256, ord("c")): 257,
        },
    )

    assert tokenizer.decode([257, ord(" "), 256]) == "abc ab"


def test_decode_raises_for_unknown_token_id() -> None:
    tokenizer = BPETokenizer(vocab=byte_vocab(), merges={})

    with pytest.raises(ValueError, match="Unknown token id"):
        tokenizer.decode([999])


def test_decode_round_trips_encoded_text() -> None:
    tokenizer = BPETokenizer.train(["ab ab"], vocab_size=257, min_frequency=2)

    token_ids = tokenizer.encode("cab")

    assert token_ids == [ord("c"), 256]
    assert tokenizer.decode(token_ids) == "cab"
