from scratch_transformer.tokenizer import BPETokenizer


def byte_vocab() -> dict[int, bytes]:
    return {i: bytes([i]) for i in range(256)}


def test_encode_without_merges_returns_utf8_byte_ids() -> None:
    tokenizer = BPETokenizer(vocab=byte_vocab(), merges={})

    assert tokenizer.encode("A\n") == [65, 10]
    assert tokenizer.encode("é") == list("é".encode("utf-8"))


def test_encode_applies_single_learned_merge() -> None:
    vocab = byte_vocab()
    vocab[256] = b"ab"
    tokenizer = BPETokenizer(vocab=vocab, merges={(ord("a"), ord("b")): 256})

    assert tokenizer.encode("ab") == [256]
    assert tokenizer.encode("cab") == [ord("c"), 256]


def test_encode_applies_merges_in_training_order() -> None:
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

    assert tokenizer.encode("abc abc") == [257, ord(" "), 257]


def test_encode_uses_train_learned_merges() -> None:
    tokenizer = BPETokenizer.train(["ab ab"], vocab_size=257, min_frequency=2)

    assert tokenizer.encode("ab") == [256]
