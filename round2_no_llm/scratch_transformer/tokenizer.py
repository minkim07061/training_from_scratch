"""Round 2 BPE tokenizer scaffold.

Goal:
    Rebuild a byte-level BPE tokenizer from memory.

Important invariants:
    - Initial token ids 0..255 represent single UTF-8 byte values.
    - Learned merge token ids start at 256.
    - Every token id appearing in the training corpus must exist in vocab.
    - vocab maps token id -> bytes.
    - merges maps pair of token ids -> new token id.
    - merge order matters during encode.

Suggested test order:
    python3 -m pytest tests/test_tokenizer_train.py
    python3 -m pytest tests/test_tokenizer_encode.py
    python3 -m pytest tests/test_tokenizer_decode.py
    python3 -m pytest tests/test_tokenizer_save_load.py
"""

from dataclasses import dataclass, field
from pathlib import Path
import json

# Returns merged pair token_ids in tuple and new updated corpus.
def single_merge(corpus, next_vocab_id, min_frequency):
    pair_counts = {}
    for i in range(1, len(corpus)):
        pair = (corpus[i-1], corpus[i])
        if pair not in pair_counts:
            pair_counts[pair] = 1
        else:
            pair_counts[pair] = pair_counts[pair] + 1
    pair_counts = dict(sorted(pair_counts.items(), key=lambda item: item[1], reverse=True))
    if not pair_counts or list(pair_counts.items())[0][1] < min_frequency:
        return corpus, None
    max_count_pair = list(pair_counts.items())[0][0]
    new_corpus = []
    curr_index = 0 
    while curr_index < len(corpus):
        if curr_index == len(corpus)-1:
            new_corpus.append(corpus[curr_index])
        elif (corpus[curr_index], corpus[curr_index+1]) == max_count_pair:
            new_corpus.append(next_vocab_id)
            curr_index += 1
        else:
            new_corpus.append(corpus[curr_index])
        curr_index += 1
    return new_corpus, max_count_pair

@dataclass
class BPETokenizer:
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
        """Learn byte-level BPE merges from raw text.

        Implementation checkpoints:
            1. Build initial vocab for all 256 byte values.
            2. Convert each input string to UTF-8 byte token ids.
            3. Count adjacent pairs in the current corpus.
            4. Stop when there are no pairs or best count < min_frequency.
            5. Assign the next token id to the best pair.
            6. Store the new token bytes by concatenating vocab[left] + vocab[right].
            7. Replace all non-overlapping occurrences of that pair in the corpus
               with the new token id.
            8. Repeat until len(vocab) reaches vocab_size.

        Common bugs to avoid:
            - Do not use bytes([token_id]) for learned ids above 255.
            - Do not replace a merged pair with left + right numerically.
            - Do not use a for-loop if you need to skip over a matched pair.
        """
        vocab = {}
        curr_vocab_id = 0
        # Create the initial vocab where the id is int(byte) and value is bytes
        for i in range(256):
            vocab[curr_vocab_id] = bytes([i])
            curr_vocab_id += 1
        
        # Corpus stores the list of token_ids
        corpus = []
        for text in texts:
            for byte in list(bytes(text.encode("utf-8"))):
                corpus.append(byte)
        
        # Merge until the vocab size limit reaches.
        merges = {}
        while curr_vocab_id < vocab_size:
            corpus, merged_pair = single_merge(corpus, curr_vocab_id, min_frequency)
            if merged_pair is None:
                print("No more merge is possible")
                break
            vocab[curr_vocab_id] = vocab[merged_pair[0]] + vocab[merged_pair[1]] # This concatenates the bytes not doing arithmetic sum.
            merges[merged_pair] = curr_vocab_id
            curr_vocab_id += 1 
        
        return cls(vocab=vocab, merges=merges, unk_token_id=self.unk_token_id)
                
        

    def encode(self, text: str) -> list[int]:
        """Convert text to token ids.

        Implementation checkpoints:
            1. Start from list(text.encode("utf-8")).
            2. For each learned merge in insertion order, replace that pair.
            3. Return the final token id list.

        Common bugs to avoid:
            - Do not drop the final token when no pair follows it.
            - Do not check token_ids[i - 1], token_ids[i] when you mean i, i + 1.
        """
        token_ids = []
        for byte in list(text.encode("utf-8")):
            token_ids.append(int(byte))
        
        for pair, merged_token_id in self.merges.items():
            new_token_ids = []
            i = 0
            while i < len(token_ids):
                if i == len(token_ids)-1:
                    new_token_ids.append(token_ids[i])
                elif pair == (token_ids[i], token_ids[i+1]):
                    new_token_ids.append(merged_token_id)
                    i += 1
                else:
                    new_token_ids.append(token_ids[i])
                i += 1
            token_ids = new_token_ids
        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        """Convert token ids back to text.

        Implementation checkpoints:
            1. For each id, verify it exists in vocab.
            2. Accumulate the bytes for every token.
            3. Join all byte pieces.
            4. Decode as UTF-8.

        Chosen behavior:
            - Unknown token id should raise ValueError.
        """
        bytes_list = []
        for token_id in token_ids:
            if token_id not in self.vocab:
                raise ValueError("Unknown token id")
            bytes_list.extend(list(self.vocab[token_id]))
        return bytes(bytes_list).decode("utf-8")

    def save(self, path: str | Path) -> None:
        """Save tokenizer state to JSON-friendly data.

        JSON conversion reminders:
            - int keys become string keys.
            - bytes become list[int].
            - tuple merge keys should become ordered list entries.
            - preserve merge order.
        """
        vocab = {}
        for vocab_id, vocab_bytes in self.vocab.items():
            vocab[str(vocab_id)] = list(vocab_bytes)
        
        merges = {}
        for merge_pair_id, new_id in self.merges.items():
            merges[str(merge_pair_id[0]) + ":" + str(merge_pair_id[1])] = str(new_id)

        json_dict = {}
        json_dict["vocab"] = vocab
        json_dict["merges"] = merges
        if self.unk_token_id:
            json_dict["unk_token_id"] = str(self.unk_token_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(json_dict, f) 

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        """Load tokenizer state saved by save."""
        with open(path, "r", encoding="utf-8") as f:
            json_dict = json.load(f)
        json_vocab = json_dict["vocab"]
        vocab = {}
        for str_id, bytes_list in json_vocab.items():
            vocab[int(str_id)] = bytes(bytes_list)
        json_merges = json_dict["merges"]
        merges = {}
        for merges_pair, merge_id in json_merges.items():
            merges_pair = merges_pair.split(":")
            merges[(int(merges_pair[0]), int(merges_pair[1]))] = int(merge_id)
        unk_token_id = None
        if "unk_token_id" in json_dict:
            unk_token_id = int(json_dict["unk_token_id"])
        return cls(vocab=vocab, merges=merges, unk_token_id=unk_token_id)  

