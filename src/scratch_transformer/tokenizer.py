"""Byte pair encoding tokenizer scaffold.

Implement this file before training a model. A BPE tokenizer usually has three
phases:

1. Convert raw text to an initial sequence of byte or character tokens.
2. Learn merge rules from a corpus.
3. Apply the learned merges during encode, then invert them during decode.
"""

from dataclasses import dataclass, field
from pathlib import Path
from collections import Counter
import os
import json


def single_merge(corpus, min_frequency, next_token_id):
    pair_counts = Counter()
    for i in range(1, len(corpus)):
        pair_counts[(corpus[i-1], corpus[i])] += 1
    merged_pair, pair_count = pair_counts.most_common(1)[0]
    if pair_count < min_frequency:
        return corpus, None
    
    new_corpus = list()
    i = 0
    while i < len(corpus)-1:
        if merged_pair == (corpus[i], corpus[i+1]):
            new_corpus.append(next_token_id)
            i += 1
        else:
            new_corpus.append(corpus[i])
        i += 1
    
    return new_corpus, merged_pair

def try_merge(token_ids, pair, merge_token_id):
    merged_tokens = []
    i = 0
    while i < len(token_ids):
        if i == len(token_ids) -1:
            merged_tokens.append(token_ids[i])
        elif (token_ids[i], token_ids[i+1]) == pair:
            merged_tokens.append(merge_token_id)
            i += 1
        else:
            merged_tokens.append(token_ids[i])
        i += 1
    return merged_tokens

@dataclass
class BPETokenizer:
    """Minimal BPE tokenizer interface for the rest of the project."""

    vocab: dict[int, bytes] = field(default_factory=dict)
    merges: dict[tuple[int, int], int] = field(default_factory=dict)
    unk_token_id: int | None = None

    @classmethod
    def train(
        cls,
        texts: list[str],
        vocab_size: int,
        *,
        min_frequency: int = 2,
    ) -> "BPETokenizer":
        """Learn BPE merges from text and return a tokenizer.

        TODO:
        - Start with byte-level tokens so every UTF-8 string is representable.
        - Count adjacent token-pair frequencies across the training corpus.
        - Repeatedly merge the most frequent pair until vocab_size is reached.
        - Store both the final vocab and pair -> new-token-id merge table.
        """
        vocab = {}
        merges = {}
        # Add initial byte tokens to vocab.
        next_token_id = 0
        for i in range(256):
            vocab[next_token_id] = bytes([i])
            next_token_id += 1

        # create the initial corpus
        corpus = []
        for text in texts:
            byte_tokens = list(text.encode('utf-8'))
            corpus.extend(byte_tokens)
        while len(vocab) < vocab_size:
            corpus, merged_pair = single_merge(corpus, min_frequency, next_token_id)
            if not merged_pair:
                print("Vocab size limited because no additional pair was found above min_frequency")
                break
            vocab[next_token_id] = vocab[merged_pair[0]] + vocab[merged_pair[1]]
            merges[merged_pair] = next_token_id
            next_token_id += 1
        return cls(vocab=vocab, merges=merges)

    def encode(self, text: str) -> list[int]:
        """Convert text into token ids using the learned BPE merges.

        TODO:
        - Convert text to UTF-8 bytes or initial character tokens.
        - Apply learned merges in the same priority order used during training.
        - Return a flat list of token ids.
        """
        token_ids = list(text.encode("utf-8"))
        for pair, merge_token_id in self.merges.items():
            token_ids = try_merge(token_ids, pair, merge_token_id)
        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        """Convert token ids back into a UTF-8 string.

        TODO:
        - Map each token id back to its byte sequence.
        - Concatenate bytes and decode as UTF-8.
        - Decide how to handle invalid ids or invalid UTF-8.
        """
        token_bytes_list = []
        for token_id in token_ids:
            if token_id in self.vocab:
                token_bytes = self.vocab[token_id]
                token_bytes_list.extend(token_bytes)
            else:
                raise ValueError(f"Unknown token id %d", token_id)
        return bytes(token_bytes_list).decode("utf-8")

    def save(self, path: str | Path) -> None:
        """Persist vocab and merges to disk.

        TODO:
        - Use a structured format such as JSON.
        - Convert tuple merge keys to strings because JSON keys must be strings.
        - Include enough metadata to load this tokenizer later.
        """
        # serialize vocab
        vocab_int = {}
        for vocab_id, vocab_bytes in self.vocab.items():
            vocab_int[vocab_id] = list(vocab_bytes)
        merges_no_tuple = []
        for pair, token_id in self.merges.items():
            merges_no_tuple.append([list(pair), token_id])
        data = {"vocab": vocab_int, "merges": merges_no_tuple, "unk_token_id": self.unk_token_id}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        """Load a tokenizer saved by :meth:`save`."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        vocab_int = data["vocab"]
        merges_no_tuple = data["merges"]
        unk_token_id = data["unk_token_id"]
        vocab = {}
        for vocab_id, vocab_int in vocab_int.items():
            vocab[int(vocab_id)] = bytes(vocab_int)
        merges = {}
        for pair_list, token_id in merges_no_tuple:
            pair = tuple(pair_list)
            merges[pair] = int(token_id)
        return cls(vocab=vocab, merges=merges, unk_token_id=unk_token_id)



