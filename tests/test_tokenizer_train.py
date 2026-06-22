from scratch_transformer.tokenizer import BPETokenizer


def test_train_initializes_byte_vocab() -> None:
    tokenizer = BPETokenizer.train([], vocab_size=256)

    assert isinstance(tokenizer, BPETokenizer)
    assert len(tokenizer.vocab) == 256
    assert tokenizer.vocab[0] == b"\x00"
    assert tokenizer.vocab[10] == b"\n"
    assert tokenizer.vocab[32] == b" "
    assert tokenizer.vocab[97] == b"a"
    assert tokenizer.vocab[255] == b"\xff"
    assert tokenizer.merges == {}


def test_train_learns_most_frequent_adjacent_pair_as_first_merge() -> None:
    tokenizer = BPETokenizer.train(["ab ab"], vocab_size=257, min_frequency=2)

    assert len(tokenizer.vocab) == 257
    assert tokenizer.merges == {(ord("a"), ord("b")): 256}
    assert tokenizer.vocab[256] == b"ab"


def test_train_stops_when_no_pair_reaches_min_frequency() -> None:
    tokenizer = BPETokenizer.train(["abc"], vocab_size=300, min_frequency=2)

    assert len(tokenizer.vocab) == 256
    assert tokenizer.merges == {}
