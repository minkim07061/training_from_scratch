import json

from scratch_transformer.tokenizer import BPETokenizer


def byte_vocab() -> dict[int, bytes]:
    return {i: bytes([i]) for i in range(256)}


def test_save_writes_json_friendly_tokenizer_data(tmp_path) -> None:
    vocab = byte_vocab()
    vocab[256] = b"ab"
    tokenizer = BPETokenizer(
        vocab=vocab,
        merges={(ord("a"), ord("b")): 256},
        unk_token_id=None,
    )
    path = tmp_path / "tokenizer.json"

    tokenizer.save(path)

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["vocab"]["97"] == [97]
    assert data["vocab"]["256"] == [97, 98]
    assert data["merges"] == [[[97, 98], 256]]
    assert data["unk_token_id"] is None


def test_load_reconstructs_tokenizer_from_json_data(tmp_path) -> None:
    path = tmp_path / "tokenizer.json"
    path.write_text(
        json.dumps(
            {
                "vocab": {
                    "97": [97],
                    "98": [98],
                    "99": [99],
                    "256": [97, 98],
                    "257": [97, 98, 99],
                },
                "merges": [
                    [[97, 98], 256],
                    [[256, 99], 257],
                ],
                "unk_token_id": None,
            }
        ),
        encoding="utf-8",
    )

    tokenizer = BPETokenizer.load(path)

    assert tokenizer.vocab[97] == b"a"
    assert tokenizer.vocab[256] == b"ab"
    assert tokenizer.vocab[257] == b"abc"
    assert tokenizer.merges == {
        (97, 98): 256,
        (256, 99): 257,
    }
    assert tokenizer.unk_token_id is None
    assert tokenizer.encode("abc") == [257]


def test_save_then_load_round_trips_tokenizer_state(tmp_path) -> None:
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
    path = tmp_path / "tokenizer.json"

    tokenizer.save(path)
    loaded = BPETokenizer.load(path)

    assert loaded.vocab == tokenizer.vocab
    assert loaded.merges == tokenizer.merges
    assert loaded.unk_token_id == tokenizer.unk_token_id
    assert loaded.encode("abc abc") == [257, ord(" "), 257]
