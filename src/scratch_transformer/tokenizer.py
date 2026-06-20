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


# input: corpus of tokens
# locally: map of pair counts
# output: most common pair, updated corpus with tokens merged
def single_merge(corpus, min_frequency):
    pair_counts = Counter((bytes, bytes))
    for i in range(1, len(corpus)):
        pair_counts[(corpus[i-1], corpus[i])] += 1
    merged_pair, pair_count = pair_counts.most_common(1)
    if pair_count < min_frequency:
        return corpus, None
    
    new_corpus = list()
    for i in range(len(corpus)-1):
        if pair == (corpus[i], corpus[i+1]):
            new_corpus.append(corpus[i] + corpus[i+1])
            i += 1
        else:
            new_corpus.append(corpus[i])
    
    return new_corpus, merged_pair

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
        # Add initial byte tokens to vocab.
        next_token_id = 0
        for i in range(256):
            vocab[next_token_id] = bytes(i)
            next_token_id += 1

        # create the initial corpus
        corpus = []
        for text in texts:
            byte_tokens = list(text.encode('utf-8'))
            corpus = corpus.extend(byte_tokens)

        while len(self.vocab) < vocab_size:
            corpus, merged_pair = single_merge(corpus, min_frequency)
            if not merged_pair:
                print("Vocab size limited because no additional pair was found above min_frequency")
                break
            self.vocab[next_token_id] = merged_pair[0] + merged_pair[1]
            merges[merged_pair] = next_token_id
            next_token_id += 1

    def encode(self, text: str) -> list[int]:
        """Convert text into token ids using the learned BPE merges.

        TODO:
        - Convert text to UTF-8 bytes or initial character tokens.
        - Apply learned merges in the same priority order used during training.
        - Return a flat list of token ids.
        """
        raise NotImplementedError("TODO: implement BPE encode in tokenizer.py")

    def decode(self, token_ids: list[int]) -> str:
        """Convert token ids back into a UTF-8 string.

        TODO:
        - Map each token id back to its byte sequence.
        - Concatenate bytes and decode as UTF-8.
        - Decide how to handle invalid ids or invalid UTF-8.
        """
        raise NotImplementedError("TODO: implement BPE decode in tokenizer.py")

    def save(self, path: str | Path) -> None:
        """Persist vocab and merges to disk.

        TODO:
        - Use a structured format such as JSON.
        - Convert tuple merge keys to strings because JSON keys must be strings.
        - Include enough metadata to load this tokenizer later.
        """
        raise NotImplementedError("TODO: implement tokenizer serialization")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        """Load a tokenizer saved by :meth:`save`."""
        raise NotImplementedError("TODO: implement tokenizer deserialization")
