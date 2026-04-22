"""
bpe_tokenizer.py
================
Byte Pair Encoding (BPE) algorithm implemented from scratch in pure Python.

This module is part of the project:
    "A bilingual PT+EN LLM with BPE tokenizer and training loop
     implemented from scratch, with didactic and documented code"

Author  : André Costa
License : MIT

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THEORETICAL BACKGROUND
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

What is tokenization?
---------------------
Language models do not operate on raw characters or whole words —
they operate on *tokens*, intermediate text units. Tokenization is
the process of converting text into sequences of integers that the
model can process.

    Text  →  Tokens  →  Integer IDs  →  Embeddings  →  Model

Why not use whole words?
------------------------
Word-level vocabularies have two serious problems:

    1. Huge vocabulary: Portuguese and English together have hundreds
       of thousands of words. Each would need its own embedding —
       infeasible for small models.

    2. Unknown words (OOV - Out of Vocabulary): any word not seen
       during training produces an <UNK> token, losing semantic
       information.

Why not use individual characters?
------------------------------------
Character vocabularies solve OOV, but produce very long sequences.
The sentence "Hello world" becomes 11 tokens instead of 2.
Long sequences increase computational cost quadratically in the
Transformer attention mechanism (O(n²)).

BPE as a compromise
---------------------
Byte Pair Encoding (Gage, 1994; Sennrich et al., 2016) finds a
middle ground: it starts with characters and iteratively merges the
most frequent pairs, building a subword vocabulary.

    "learning"  → ["learn", "ing"]
    "learned"   → ["learn", "ed"]
    "learnable" → ["learn", "able"]

The prefix "learn" is shared — the model learns morphology
naturally, without explicit supervision.

References:
    - Gage, P. (1994). A new algorithm for data compression.
      C Users Journal, 12(2), 23-38.
    - Sennrich, R., Haddow, B., & Birch, A. (2016). Neural machine
      translation of rare words with subword units. ACL 2016.
    - Radford, A. et al. (2019). Language models are unsupervised
      multitask learners. (GPT-2 — popularized BPE in LLMs)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BPE ALGORITHM — OVERVIEW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Training (offline, done once on the corpus):
    1. Encode each byte of the corpus as an initial token (base vocab = 256)
    2. Count the frequency of all adjacent token pairs
    3. Select the most frequent pair (p_max)
    4. Create a new token = merge of p_max
    5. Replace all occurrences of p_max with the new token
    6. Repeat steps 2–5 until reaching the desired vocab_size

Encoding (online, for each new text):
    1. Convert text to bytes
    2. Apply learned merges in the order they were learned
    3. Return the sequence of IDs

Decoding:
    1. Convert IDs back to bytes using the vocabulary
    2. Decode the bytes as UTF-8
"""

# ─────────────────────────────────────────────────────────────
# Imports — standard Python library only, no external dependencies
# except 'regex' (better Unicode support than 're')
# ─────────────────────────────────────────────────────────────
import os
import json
import regex  # pip install regex
from collections import defaultdict
from typing import Optional


# ─────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────

def get_pairs(ids: list[int]) -> dict[tuple[int, int], int]:
    """
    Count the frequency of all adjacent pairs in a sequence.

    This is the central operation of BPE. For each position i in the
    sequence, forms the pair (ids[i], ids[i+1]) and increments its count.

    Example:
        ids = [1, 2, 3, 2, 1, 2]
        returns: {(1,2): 2, (2,3): 1, (3,2): 1, (2,1): 1}

    Complexity: O(n), where n = len(ids)

    Args:
        ids: Sequence of token IDs.

    Returns:
        Dictionary mapping each pair to its frequency.
    """
    counts: dict[tuple[int, int], int] = defaultdict(int)
    for pair in zip(ids, ids[1:]):
        counts[pair] += 1
    return counts


def merge_sequence(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """
    Replace all occurrences of `pair` in `ids` with token `new_id`.

    This function implements the "merge" step of BPE. It scans the
    sequence once from left to right, replacing each occurrence of the
    target pair with the new token.

    Example:
        ids    = [1, 2, 3, 1, 2]
        pair   = (1, 2)
        new_id = 99
        returns: [99, 3, 99]

    Note: Replacement is non-overlapping. The sequence (1,2,1,2) with
    pair=(1,2) results in [99, 99], not [1, 99, 2] or [99, 1, 2].

    Complexity: O(n), where n = len(ids)

    Args:
        ids:    Original sequence of IDs.
        pair:   Token pair to merge (a, b).
        new_id: ID of the new token resulting from the merge.

    Returns:
        New sequence with merges applied.
    """
    result: list[int] = []
    i = 0
    while i < len(ids):
        # Check whether the pair starts at position i (and is not the last element)
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            result.append(new_id)
            i += 2  # skip the two tokens that were merged
        else:
            result.append(ids[i])
            i += 1
    return result


# ─────────────────────────────────────────────────────────────
# Pre-tokenization pattern (GPT-4 / tiktoken style)
# ─────────────────────────────────────────────────────────────

# This regex pattern splits text into "words" before applying BPE.
# Pre-tokenization ensures BPE never merges tokens across word
# boundaries (e.g., the space before "hello" and the "h" in "hello"
# will never form a single token).
#
# The pattern captures, in order of priority:
#   1. English contractions: 's, 't, 're, 've, 'm, 'll, 'd
#   2. Words optionally preceded by a space
#   3. Numbers optionally preceded by a space
#   4. Non-alphanumeric characters optionally preceded by a space
#   5. Whitespace (without capturing the space that precedes words)
#
# Reference: https://github.com/openai/tiktoken
GPT4_SPLIT_PATTERN = regex.compile(
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
)


# ─────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────

class BPETokenizer:
    """
    Byte Pair Encoding (BPE) tokenizer implemented from scratch.

    This implementation operates directly on UTF-8 bytes, which guarantees:
        - Full coverage of any Unicode text (PT, EN, emojis, etc.)
        - Fixed base vocabulary of exactly 256 tokens (one per byte)
        - No <UNK> tokens — any text is encodable

    Public attributes:
        vocab_size (int): Total vocabulary size after training.
        merges (dict):    Table of learned merges. Maps
                          (id_a, id_b) → id_new.
        vocab (dict):     Full vocabulary. Maps id → bytes.

    Basic usage:
        >>> tokenizer = BPETokenizer(vocab_size=1000)
        >>> tokenizer.train(["Hello world. Olá mundo."])
        >>> ids = tokenizer.encode("Hello")
        >>> tokenizer.decode(ids)
        'Hello'
    """

    def __init__(self, vocab_size: int = 16384):
        """
        Initialize the tokenizer.

        The base vocabulary always starts with the 256 possible bytes (0–255).
        The number of merges to be learned is vocab_size - 256.

        Args:
            vocab_size: Desired final vocabulary size.
                        Typical values: 4096, 8192, 16384, 32768.
                        Must be greater than 256.

        Raises:
            ValueError: If vocab_size <= 256.
        """
        if vocab_size <= 256:
            raise ValueError(
                f"vocab_size must be greater than 256 (byte base vocabulary). "
                f"Received: {vocab_size}"
            )

        self.vocab_size: int = vocab_size

        # merges: table of merges learned during training
        # key   : (id_token_a, id_token_b)
        # value : id_token_new
        # ORDER matters — merges are applied in the order they were learned
        self.merges: dict[tuple[int, int], int] = {}

        # vocab: full dictionary id → byte sequence
        # Initialized with the 256 base bytes; expanded during training
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

        # Pre-tokenization pattern (splits text into words before BPE)
        self._split_pattern = GPT4_SPLIT_PATTERN

    # ─────────────────────────────────────────────────────────
    # Training
    # ─────────────────────────────────────────────────────────

    def train(self, corpus: list[str], verbose: bool = False) -> None:
        """
        Train the BPE tokenizer on a text corpus.

        Training executes `vocab_size - 256` merge iterations.
        In each iteration:
            1. Count all adjacent pairs in the tokenized corpus
            2. Select the most frequent pair
            3. Record the merge in self.merges
            4. Update self.vocab with the new token
            5. Apply the merge to the corpus (in-place)

        Total complexity: O(N × M), where:
            N = total number of tokens in the corpus (decreases each merge)
            M = number of merges = vocab_size - 256

        Args:
            corpus:  List of strings forming the training corpus.
                     Example: ["Text in Portuguese.", "Text in English."]
            verbose: If True, prints progress after each merge.

        Example:
            >>> tok = BPETokenizer(vocab_size=300)
            >>> tok.train(["abracadabra " * 100], verbose=True)
            Merge     1/44 | pair: (b'a', b'b') → token 256 | freq: 200
            ...
        """
        num_merges = self.vocab_size - 256

        # ── Step 1: Pre-tokenization ──────────────────────────────────────
        # Split the corpus into "words" using the regex pattern.
        # Each word is converted to its UTF-8 byte representation.
        #
        # Example:
        #   "Hello world" → ["Hello", " world"]
        #                 → [b'Hello', b' world']
        #
        # Result: list of lists of integers (byte IDs 0–255)
        ids_per_chunk: list[list[int]] = []
        for text in corpus:
            words = regex.findall(self._split_pattern, text)
            for word in words:
                word_bytes = word.encode("utf-8")
                ids_per_chunk.append(list(word_bytes))

        if verbose:
            total_tokens = sum(len(chunk) for chunk in ids_per_chunk)
            print(f"Pre-tokenization complete.")
            print(f"  Chunks (words): {len(ids_per_chunk)}")
            print(f"  Total initial tokens (bytes): {total_tokens}")
            print(f"  Merges to perform: {num_merges}\n")

        # ── Step 2: Main merge loop ───────────────────────────────────────
        for merge_idx in range(num_merges):

            # Count pairs across all corpus chunks
            pair_counts: dict[tuple[int, int], int] = defaultdict(int)
            for chunk_ids in ids_per_chunk:
                chunk_pairs = get_pairs(chunk_ids)
                for pair, count in chunk_pairs.items():
                    pair_counts[pair] += count

            # If no more pairs exist, the corpus is too small
            if not pair_counts:
                if verbose:
                    print(f"Warning: corpus exhausted after {merge_idx} merges.")
                break

            # Select the most frequent pair
            best_pair = max(pair_counts, key=lambda p: pair_counts[p])
            best_freq = pair_counts[best_pair]

            # ID of the new token = next available integer
            new_id = 256 + merge_idx

            # Record the merge
            self.merges[best_pair] = new_id

            # Update the vocabulary:
            # The new token is the concatenation of the bytes of both merged tokens
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            # Apply the merge to all corpus chunks
            ids_per_chunk = [
                merge_sequence(chunk, best_pair, new_id)
                for chunk in ids_per_chunk
            ]

            if verbose:
                token_str_a = self.vocab[best_pair[0]]
                token_str_b = self.vocab[best_pair[1]]
                print(
                    f"Merge {merge_idx + 1:>5}/{num_merges} | "
                    f"pair: ({token_str_a!r}, {token_str_b!r}) "
                    f"→ token {new_id} | "
                    f"freq: {best_freq}"
                )

        if verbose:
            total_after = sum(len(chunk) for chunk in ids_per_chunk)
            print(f"\nTraining complete.")
            print(f"  Final vocabulary: {len(self.vocab)} tokens")
            print(f"  Total tokens after merges: {total_after}")

    # ─────────────────────────────────────────────────────────
    # Encoding
    # ─────────────────────────────────────────────────────────

    def encode(self, text: str) -> list[int]:
        """
        Convert a string into a sequence of token IDs.

        The encoding process follows these steps:
            1. Split text into chunks via pre-tokenization (regex)
            2. Convert each chunk to bytes → list of IDs (0–255)
            3. Apply learned merges in order to each chunk
            4. Concatenate IDs from all chunks

        Applying merges in order is crucial: merges learned first have
        priority. This ensures consistency with training.

        Args:
            text: Text to encode. Can be any UTF-8 string.

        Returns:
            List of integers representing the tokens.

        Raises:
            RuntimeError: If the tokenizer has not been trained (empty merges).

        Example:
            >>> tok.encode("Hello")
            [323, 195]   # IDs depend on training
        """
        if not self.merges:
            raise RuntimeError(
                "The tokenizer has not been trained. "
                "Call .train() before .encode()."
            )

        all_ids: list[int] = []

        chunks = regex.findall(self._split_pattern, text)

        for chunk in chunks:
            # Convert to bytes then to list of integer IDs
            chunk_ids = list(chunk.encode("utf-8"))

            # Apply all learned merges in order
            while len(chunk_ids) >= 2:
                pairs = get_pairs(chunk_ids)

                # Find the pair with the lowest index in self.merges
                # (= pair learned first = highest priority)
                best_pair = min(
                    pairs,
                    key=lambda p: self.merges.get(p, float("inf"))
                )

                # If no pair is in merges, we are done with this chunk
                if best_pair not in self.merges:
                    break

                new_id = self.merges[best_pair]
                chunk_ids = merge_sequence(chunk_ids, best_pair, new_id)

            all_ids.extend(chunk_ids)

        return all_ids

    # ─────────────────────────────────────────────────────────
    # Decoding
    # ─────────────────────────────────────────────────────────

    def decode(self, ids: list[int]) -> str:
        """
        Convert a sequence of IDs back to a string.

        Each ID is mapped to its byte sequence via self.vocab,
        and the bytes are concatenated and decoded as UTF-8.

        Note on UTF-8 errors:
            Individual tokens may correspond to incomplete bytes
            (e.g., the first half of a 2-byte UTF-8 character).
            Therefore, we concatenate ALL bytes before decoding,
            and use errors="replace" to handle invalid sequences
            that may arise from out-of-context IDs.

        Args:
            ids: Sequence of IDs to decode.

        Returns:
            Decoded string.

        Example:
            >>> tok.decode([323, 195])
            'Hello'
        """
        raw_bytes = b"".join(self.vocab[i] for i in ids)
        return raw_bytes.decode("utf-8", errors="replace")

    # ─────────────────────────────────────────────────────────
    # Persistence (save / load)
    # ─────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """
        Save the trained tokenizer to disk.

        Creates two files in directory `path`:
            tokenizer.json  — metadata and merge table (human-readable)
            vocab.json      — full vocabulary id → byte representation

        JSON format was chosen for being readable, portable and compatible
        with the HuggingFace ecosystem (tokenizers library).

        Structure of tokenizer.json:
            {
                "vocab_size": int,
                "num_merges": int,
                "merges": [[id_a, id_b, id_new], ...]
            }

        Args:
            path: Directory path where files will be saved.
                  Created if it does not exist.
        """
        os.makedirs(path, exist_ok=True)

        merges_list = [
            [int(a), int(b), int(new_id)]
            for (a, b), new_id in self.merges.items()
        ]

        tokenizer_data = {
            "vocab_size": self.vocab_size,
            "num_merges": len(self.merges),
            "merges": merges_list,
        }

        with open(os.path.join(path, "tokenizer.json"), "w", encoding="utf-8") as f:
            json.dump(tokenizer_data, f, indent=2, ensure_ascii=False)

        vocab_data = {
            str(token_id): list(token_bytes)
            for token_id, token_bytes in self.vocab.items()
        }

        with open(os.path.join(path, "vocab.json"), "w", encoding="utf-8") as f:
            json.dump(vocab_data, f, indent=2, ensure_ascii=False)

        print(f"Tokenizer saved to '{path}/'")
        print(f"  tokenizer.json — {len(self.merges)} merges")
        print(f"  vocab.json     — {len(self.vocab)} tokens")

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        """
        Load a previously saved tokenizer.

        Class method (factory method): creates a new instance and fills
        it with data loaded from disk, without needing to re-train.

        Args:
            path: Directory where files were saved by .save().

        Returns:
            Ready-to-use BPETokenizer instance.

        Raises:
            FileNotFoundError: If files do not exist at the given path.

        Example:
            >>> tok = BPETokenizer.load("./my_tokenizer")
            >>> tok.encode("Hello world")
        """
        tokenizer_path = os.path.join(path, "tokenizer.json")
        vocab_path     = os.path.join(path, "vocab.json")

        with open(tokenizer_path, "r", encoding="utf-8") as f:
            tokenizer_data = json.load(f)

        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)

        tokenizer = cls(vocab_size=tokenizer_data["vocab_size"])

        for a, b, new_id in tokenizer_data["merges"]:
            tokenizer.merges[(int(a), int(b))] = int(new_id)

        tokenizer.vocab = {
            int(token_id): bytes(token_bytes)
            for token_id, token_bytes in vocab_data.items()
        }

        print(f"Tokenizer loaded from '{path}/'")
        print(f"  vocab_size : {tokenizer.vocab_size}")
        print(f"  merges     : {len(tokenizer.merges)}")

        return tokenizer

    # ─────────────────────────────────────────────────────────
    # Utilities and inspection
    # ─────────────────────────────────────────────────────────

    def token_to_str(self, token_id: int) -> str:
        """
        Return the human-readable representation of a token by its ID.

        Useful for inspecting the vocabulary and understanding which
        subwords the tokenizer has learned.

        Args:
            token_id: ID of the token to inspect.

        Returns:
            String representing the token bytes (decoded if possible).
        """
        token_bytes = self.vocab.get(token_id, b"<unknown>")
        try:
            return token_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return repr(token_bytes)

    def vocab_stats(self) -> None:
        """
        Print statistics about the trained vocabulary.

        Displays the 20 longest learned tokens, which generally
        correspond to words or subwords that are very frequent in the corpus.
        """
        print(f"\n{'='*50}")
        print(f"  BPE Vocabulary Statistics")
        print(f"{'='*50}")
        print(f"  vocab_size  : {self.vocab_size}")
        print(f"  base tokens : 256 (bytes 0–255)")
        print(f"  merges      : {len(self.merges)}")
        print(f"\n  20 longest tokens (frequent subwords):")

        sorted_vocab = sorted(
            [(tid, tb) for tid, tb in self.vocab.items() if tid >= 256],
            key=lambda x: len(x[1]),
            reverse=True
        )

        for token_id, token_bytes in sorted_vocab[:20]:
            try:
                readable = token_bytes.decode("utf-8")
            except UnicodeDecodeError:
                readable = repr(token_bytes)
            print(f"    [{token_id:>6}] {repr(readable):<30} ({len(token_bytes)} bytes)")

        print(f"{'='*50}\n")

    def __repr__(self) -> str:
        status = "trained" if self.merges else "not trained"
        return (
            f"BPETokenizer("
            f"vocab_size={self.vocab_size}, "
            f"merges={len(self.merges)}, "
            f"status='{status}')"
        )


# ─────────────────────────────────────────────────────────────
# Demo / quick test
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BPE Tokenizer — train and validate")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run a quick demo with a small vocab (320 tokens). "
             "Does NOT produce a tokenizer suitable for training."
    )
    args = parser.parse_args()

    # ── Demo mode (--demo flag) ───────────────────────────────────────────
    # Trains on a tiny built-in corpus with vocab_size=320.
    # Useful for understanding how BPE works, but the resulting
    # tokenizer is NOT saved to ./tokenizer and cannot be used
    # by data_pipeline.py.
    if args.demo:
        print("=" * 60)
        print("  BPETokenizer — Demo mode (vocab_size=320)")
        print("  NOTE: this tokenizer is for illustration only.")
        print("        Run without --demo to produce the real tokenizer.")
        print("=" * 60)

        corpus_demo = [
            # Portuguese
            "aprendizado de máquina é fascinante. "
            "redes neurais aprendem padrões complexos. "
            "o modelo aprende a linguagem naturalmente. "
            "aprender, aprendendo, aprendizado, aprendiz. ",
            # English
            "machine learning is fascinating. "
            "neural networks learn complex patterns. "
            "the model learns language naturally. "
            "learn, learning, learned, learner. ",
        ] * 50

        tokenizer = BPETokenizer(vocab_size=320)
        tokenizer.train(corpus_demo, verbose=True)
        tokenizer.vocab_stats()

        tests = [
            "aprendizado", "learning",
            "redes neurais", "neural networks",
            "Olá, mundo!", "Hello, world!",
        ]

        print("Encode/decode tests:")
        print("-" * 50)
        for text in tests:
            ids     = tokenizer.encode(text)
            decoded = tokenizer.decode(ids)
            tokens  = [tokenizer.token_to_str(i) for i in ids]
            print(f"  Text    : {repr(text)}")
            print(f"  IDs     : {ids}")
            print(f"  Tokens  : {tokens}")
            print(f"  Decoded : {repr(decoded)}")
            print(f"  OK      : {text == decoded}")
            print()

        print("Demo complete. No files were saved.")
        print("Run 'python bpe_tokenizer.py' (without --demo) to train the real tokenizer.")

    # ── Production mode (default) ─────────────────────────────────────────
    # Trains on a representative bilingual corpus with vocab_size=16384.
    # Saves the tokenizer to ./tokenizer/, which is the path expected
    # by data_pipeline.py.
    else:
        print("=" * 60)
        print("  BPETokenizer — Training (vocab_size=16384)")
        print("  Output: ./tokenizer/")
        print("=" * 60)

        corpus_production = [
            # Portuguese — representative sample
            "aprendizado de máquina é fascinante. "
            "redes neurais aprendem padrões complexos. "
            "o modelo aprende a linguagem naturalmente. "
            "aprender, aprendendo, aprendizado, aprendiz. "
            "o brasil é um país de dimensões continentais. "
            "a língua portuguesa é falada em vários países. "
            "ciência de dados e inteligência artificial. "
            "processamento de linguagem natural em português. ",
            # English — representative sample
            "machine learning is fascinating. "
            "neural networks learn complex patterns. "
            "the model learns language naturally. "
            "learn, learning, learned, learner. "
            "artificial intelligence and data science. "
            "natural language processing and transformers. "
            "deep learning models require large datasets. "
            "the quick brown fox jumps over the lazy dog. ",
        ] * 500  # repeated to build sufficient frequency for 16k merges

        tokenizer = BPETokenizer(vocab_size=16384)
        tokenizer.train(corpus_production, verbose=True)
        tokenizer.vocab_stats()

        # Save to ./tokenizer — the path expected by data_pipeline.py
        tokenizer.save("./tokenizer")

        # Validate save/load round-trip
        print("\nValidating save/load round-trip...")
        tokenizer2 = BPETokenizer.load("./tokenizer")

        for text in ["machine learning", "aprendizado de máquina", "Olá mundo!"]:
            ids     = tokenizer2.encode(text)
            decoded = tokenizer2.decode(ids)
            status  = "OK" if decoded == text else "FAIL"
            print(f"  [{status}] {repr(text)} → {ids[:5]}{'...' if len(ids) > 5 else ''} → {repr(decoded)}")

        print("\nTokenizer ready. You can now run:")
        print("  python data_pipeline.py --dry-run")
