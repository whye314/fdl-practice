"""
One-time data preparation.
Uses the local Nemotron-CC dataset and trains a BPE tokenizer on it.

Usage:
    python prepare.py                  # train the tokenizer (data is already local)

Tokenizer is stored in ~/.cache/gpt/ by default. Override with the
GPT_CACHE_DIR or GPT_TOKENIZER_DIR environment variables.
"""

import os
import sys
import time
import math
import random
import argparse
import pickle
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq
import rustbpe
import tiktoken
import torch

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

MAX_SEQ_LEN = 2048       # context length
EVAL_TOKENS = 40 * 524288  # number of tokens for val eval

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_DIR = os.environ.get("GPT_CACHE_DIR", os.path.join(os.path.expanduser("~"), ".cache", "gpt"))
TOKENIZER_DIR = os.environ.get("GPT_TOKENIZER_DIR", os.path.join(CACHE_DIR, "tokenizer"))

# Dataset is expected to be in the following directory:
DATA_DIR = os.path.join(os.path.expanduser("~"), "data", "nemotron_cc")
TRAIN_FILENAME = "train.parquet"
VAL_FILENAME = "validation.parquet"
VOCAB_SIZE = 8192

# Defaults for the training scripts' --data-dir/--val-file (train.py, train_singlegpu.py,
# train_phase2.py). Derived from DATA_DIR so the personal path is spelled out once.
DEFAULT_TRAIN_DATA_DIR = os.path.join(os.path.dirname(DATA_DIR), "nemotron_cc_5b")
DEFAULT_VAL_FILE = os.path.join(DATA_DIR, VAL_FILENAME)

# BPE split pattern (GPT-4 style, with \p{N}{1,2} instead of {1,3})
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

SPECIAL_TOKENS = [f"<|reserved_{i}|>" for i in range(4)]
BOS_TOKEN = "<|reserved_0|>"

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def check_data():
    """Verify the Nemotron-CC train/validation parquet files are present locally."""
    missing = [f for f in (TRAIN_FILENAME, VAL_FILENAME) if not os.path.exists(os.path.join(DATA_DIR, f))]
    if missing:
        print(f"Data: missing {missing} in {DATA_DIR}. Run download_nemotron_cc.py in the "
              f"deep_learning project first.")
        sys.exit(1)
    print(f"Data: found {TRAIN_FILENAME} and {VAL_FILENAME} at {DATA_DIR}")

# ---------------------------------------------------------------------------
# Tokenizer training
# ---------------------------------------------------------------------------

def list_parquet_files():
    """Return the Nemotron-CC train and validation parquet file paths."""
    return [os.path.join(DATA_DIR, TRAIN_FILENAME), os.path.join(DATA_DIR, VAL_FILENAME)]


def text_iterator(max_chars=1_000_000_000, doc_cap=10_000):
    """Yield documents from training split (all shards except pinned val shard)."""
    parquet_paths = [p for p in list_parquet_files() if not p.endswith(VAL_FILENAME)]
    nchars = 0
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(pf.num_row_groups):
            rg = pf.read_row_group(rg_idx)
            for text in rg.column("text").to_pylist():
                doc = text[:doc_cap] if len(text) > doc_cap else text
                nchars += len(doc)
                yield doc
                if nchars >= max_chars:
                    return


def train_tokenizer():
    """Train BPE tokenizer using rustbpe, save as tiktoken pickle."""
    tokenizer_pkl = os.path.join(TOKENIZER_DIR, "tokenizer.pkl")
    token_bytes_path = os.path.join(TOKENIZER_DIR, "token_bytes.pt")

    if os.path.exists(tokenizer_pkl) and os.path.exists(token_bytes_path):
        print(f"Tokenizer: already trained at {TOKENIZER_DIR}")
        return

    os.makedirs(TOKENIZER_DIR, exist_ok=True)

    check_data()

    # --- Train with rustbpe ---
    print("Tokenizer: training BPE tokenizer...")
    t0 = time.time()

    tokenizer = rustbpe.Tokenizer()
    vocab_size_no_special = VOCAB_SIZE - len(SPECIAL_TOKENS)
    tokenizer.train_from_iterator(text_iterator(), vocab_size_no_special, pattern=SPLIT_PATTERN)

    # Build tiktoken encoding from trained merges
    pattern = tokenizer.get_pattern()
    mergeable_ranks = {bytes(k): v for k, v in tokenizer.get_mergeable_ranks()}
    tokens_offset = len(mergeable_ranks)
    special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
    enc = tiktoken.Encoding(
        name="rustbpe",
        pat_str=pattern,
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )

    # Save tokenizer
    with open(tokenizer_pkl, "wb") as f:
        pickle.dump(enc, f)

    t1 = time.time()
    print(f"Tokenizer: trained in {t1 - t0:.1f}s, saved to {tokenizer_pkl}")

    # --- Build token_bytes lookup for BPB evaluation ---
    print("Tokenizer: building token_bytes lookup...")
    special_set = set(SPECIAL_TOKENS)
    token_bytes_list = []
    for token_id in range(enc.n_vocab):
        token_str = enc.decode([token_id])
        if token_str in special_set:
            token_bytes_list.append(0)
        else:
            token_bytes_list.append(len(token_str.encode("utf-8")))
    token_bytes_tensor = torch.tensor(token_bytes_list, dtype=torch.int32)
    torch.save(token_bytes_tensor, token_bytes_path)
    print(f"Tokenizer: saved token_bytes to {token_bytes_path}")

    # Sanity check
    test = "Hello world! Numbers: 123. Unicode: 你好"
    encoded = enc.encode_ordinary(test)
    decoded = enc.decode(encoded)
    assert decoded == test, f"Tokenizer roundtrip failed: {test!r} -> {decoded!r}"
    print(f"Tokenizer: sanity check passed (vocab_size={enc.n_vocab})")

# ---------------------------------------------------------------------------
# Runtime utilities (imported by train.py, train_singlegpu.py, train_phase2.py)
# ---------------------------------------------------------------------------

class Tokenizer:
    """Minimal tokenizer wrapper. Training is handled above."""

    def __init__(self, enc):
        self.enc = enc
        self.bos_token_id = enc.encode_single_token(BOS_TOKEN)

    @classmethod
    def from_directory(cls, tokenizer_dir=TOKENIZER_DIR):
        with open(os.path.join(tokenizer_dir, "tokenizer.pkl"), "rb") as f:
            enc = pickle.load(f)
        return cls(enc)

    def get_vocab_size(self):
        return self.enc.n_vocab

    def get_bos_token_id(self):
        return self.bos_token_id

    def encode(self, text, prepend=None, num_threads=8):
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.enc.encode_single_token(prepend)
        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend_id)
        elif isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None:
                for row in ids:
                    row.insert(0, prepend_id)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")
        return ids

    def decode(self, ids):
        return self.enc.decode(ids)


def get_token_bytes(device="cpu"):
    path = os.path.join(TOKENIZER_DIR, "token_bytes.pt")
    with open(path, "rb") as f:
        return torch.load(f, map_location=device)


def parquet_token_count(path):
    """Read only the compact num_tokens column written by the extractor."""
    column = pq.read_table(path, columns=["num_tokens"])["num_tokens"]
    return int(pc.sum(column).as_py())


def select_training_shards(data_dir, fraction, seed, log=print):
    """Select approximately `fraction` tokens independently in every stratum below
    data_dir. Falls back to a single legacy train.parquet file (prepare.py's own
    single-file layout) if no per-stratum subdirectories are found."""
    data_dir = Path(data_dir)
    stratum_dirs = sorted(path for path in data_dir.iterdir() if path.is_dir())
    if not stratum_dirs:
        legacy_train_file = data_dir / TRAIN_FILENAME
        if legacy_train_file.is_file():
            log(f"Data legacy: using {legacy_train_file}")
            return [str(legacy_train_file)], None
        raise FileNotFoundError(f"No stratum directories found below {data_dir}")

    selected = []
    selected_tokens = 0
    for stratum_dir in stratum_dirs:
        weighted = [(path, parquet_token_count(path)) for path in sorted(stratum_dir.glob("*.parquet"))]
        if not weighted:
            continue
        stable_seed = seed + sum((index + 1) * ord(char) for index, char in enumerate(stratum_dir.name))
        random.Random(stable_seed).shuffle(weighted)
        target = sum(tokens for _, tokens in weighted) * fraction
        stratum_selected = []
        stratum_tokens = 0
        for path, tokens in weighted:
            # Stop on whichever side of the target is closer. This avoids a
            # systematic one-full-shard overshoot for small strata.
            if stratum_selected and abs(stratum_tokens - target) <= abs(stratum_tokens + tokens - target):
                break
            stratum_selected.append(path)
            stratum_tokens += tokens
        selected.extend(stratum_selected)
        selected_tokens += stratum_tokens
        log(
            f"Data {stratum_dir.name}: selected {len(stratum_selected)}/{len(weighted)} shards, "
            f"{stratum_tokens:,} tokens"
        )

    if not selected:
        raise FileNotFoundError(f"No Parquet shards found below {data_dir}")
    random.Random(seed).shuffle(selected)
    return [str(path) for path in selected], selected_tokens


def _parquet_batches_sharded(parquet_paths, tokenizer_batch_size, rank, world_size,
                              columns=("text",), extract=None):
    """Yield disjoint document batches from one collection of Parquet files.
    `columns` selects which Parquet columns to read; `extract(record_batch) -> list`
    turns a batch into per-row items (default: the first column as raw text, for
    prepare.py's own tokenize-from-text path). Pass pretokenized-data variants
    (e.g. train_phase2.py's input_ids/loss_mask columns) via these two args
    instead of duplicating this function."""
    if extract is None:
        extract = lambda record_batch: record_batch.column(0).to_pylist()
    epoch = 1
    while True:
        chunk_idx = 0
        for filepath in parquet_paths:
            pf = pq.ParquetFile(filepath)
            for record_batch in pf.iter_batches(
                batch_size=tokenizer_batch_size, columns=list(columns)
            ):
                if chunk_idx % world_size == rank:
                    yield extract(record_batch), epoch
                chunk_idx += 1
        epoch += 1


def _document_batches(split, train_paths, val_path, tokenizer_batch_size=128, rank=0, world_size=1,
                       log=print, columns=("text",), extract=None, val_paths=None):
    """Yield rank-disjoint batches, interleaving training strata smoothly.
    rank=0, world_size=1 (the defaults) give a single, unsharded stream.
    `columns`/`extract` are forwarded to _parquet_batches_sharded (see there) so
    callers with a different Parquet schema (e.g. pretokenized chat data) can reuse
    this interleaving logic instead of duplicating it. `val_paths` overrides the
    non-train source files (defaults to [val_path]; useful for evaluating against
    a file other than the pinned validation set)."""
    if split != "train":
        yield from _parquet_batches_sharded(
            val_paths if val_paths is not None else [val_path],
            tokenizer_batch_size, rank, world_size, columns=columns, extract=extract,
        )
        return

    assert train_paths, "No training shards found."
    paths_by_stratum = {}
    for filepath in train_paths:
        path = Path(filepath)
        paths_by_stratum.setdefault(path.parent.name, []).append(filepath)

    # A legacy single-file dataset has no strata to interleave.
    if len(paths_by_stratum) == 1:
        yield from _parquet_batches_sharded(
            train_paths, tokenizer_batch_size, rank, world_size, columns=columns, extract=extract,
        )
        return

    names = sorted(paths_by_stratum)
    iterators = {
        name: _parquet_batches_sharded(
            sorted(paths_by_stratum[name]), tokenizer_batch_size, rank, world_size,
            columns=columns, extract=extract,
        )
        for name in names
    }
    document_counts = {
        name: sum(pq.ParquetFile(path).metadata.num_rows for path in paths_by_stratum[name])
        for name in names
    }
    total_documents = sum(document_counts.values())
    target_fractions = {
        name: document_counts[name] / total_documents for name in names
    }
    batches_drawn = {name: 0 for name in names}
    stratum_epochs = {name: 1 for name in names}
    total_batches = 0

    log(
        "Interleaving document batches across strata: "
        + ", ".join(f"{name}={target_fractions[name]:.2%}" for name in names)
    )
    while True:
        # Smooth weighted round-robin keeps the mixture stable without random
        # runs of one stratum. Weighting by documents preserves the extracted
        # token mixture because each draw contains the same number of documents.
        name = max(
            names,
            key=lambda candidate: (
                target_fractions[candidate] * (total_batches + 1)
                - batches_drawn[candidate],
                candidate,
            ),
        )
        batch, stratum_epoch = next(iterators[name])
        batches_drawn[name] += 1
        total_batches += 1
        stratum_epochs[name] = stratum_epoch
        yield batch, min(stratum_epochs.values())


def make_dataloader(tokenizer, B, T, split, train_paths, val_path, rank=0, world_size=1,
                     buffer_size=1000, log=print):
    """
    BOS-aligned dataloader with best-fit packing, sourced from a rank-sharded document
    stream (rank=0, world_size=1 for single-GPU use). Every row starts with BOS.
    Documents packed using best-fit to minimize cropping; when no document fits the
    remaining space, the shortest doc is cropped to fill exactly and its cropped
    continuation is kept for a later row. 100% utilization (no padding).
    """
    assert split in ["train", "val"]
    row_capacity = T + 1
    batches = _document_batches(split, train_paths, val_path, 128, rank, world_size, log)
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer = []
    epoch = 1

    def refill_buffer():
        nonlocal epoch
        doc_batch, epoch = next(batches)
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token)
        doc_buffer.extend(token_lists)

    # Pre-allocate buffers: [inputs (B*T) | targets (B*T)]
    row_buffer = torch.empty((B, row_capacity), dtype=torch.long)
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=True)
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device="cuda")
    cpu_inputs = cpu_buffer[:B * T].view(B, T)
    cpu_targets = cpu_buffer[B * T:].view(B, T)
    inputs = gpu_buffer[:B * T].view(B, T)
    targets = gpu_buffer[B * T:].view(B, T)

    while True:
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                while len(doc_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - pos

                # Find largest doc that fits entirely
                best_idx = -1
                best_len = 0
                for i, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len

                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    row_buffer[row_idx, pos:pos + len(doc)] = torch.tensor(doc, dtype=torch.long)
                    pos += len(doc)
                else:
                    # No doc fits — crop shortest to fill remaining, keep the tail
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos:pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                    if len(doc) > remaining:
                        doc_buffer.append(doc[remaining:])
                    pos += remaining

        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])
        gpu_buffer.copy_(cpu_buffer, non_blocking=True)
        yield inputs, targets, epoch

# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_bpb(model, tokenizer, batch_size, val_path=None, rank=0, world_size=1):
    """
    Bits per byte (BPB): vocab size-independent evaluation metric.
    Sums per-token cross-entropy (in nats), sums target byte lengths,
    then converts nats/byte to bits/byte. Special tokens (byte length 0)
    are excluded from both sums.
    Uses fixed MAX_SEQ_LEN so results are comparable across configs.
    Defaults to the canonical DEFAULT_VAL_FILE so BPB stays a fixed, comparable
    benchmark: it does NOT follow a training script's own --val-file (which may
    point at a different, differently-formatted dataset, e.g. phase2's pretokenized
    chat data). Only pass val_path explicitly if you deliberately want to score
    against a different raw-text Parquet file.
    """
    if val_path is None:
        val_path = DEFAULT_VAL_FILE
    token_bytes = get_token_bytes(device="cuda")
    val_loader = make_dataloader(tokenizer, batch_size, MAX_SEQ_LEN, "val",
                                  train_paths=None, val_path=val_path,
                                  rank=rank, world_size=world_size)
    steps = EVAL_TOKENS // (batch_size * MAX_SEQ_LEN)
    total_nats = 0.0
    total_bytes = 0
    for _ in range(steps):
        x, y, _ = next(val_loader)
        loss_flat = model(x, y, reduction='none').view(-1)
        y_flat = y.view(-1)
        nbytes = token_bytes[y_flat]
        mask = nbytes > 0
        total_nats += (loss_flat * mask).sum().item()
        total_bytes += nbytes.sum().item()
    return total_nats / (math.log(2) * total_bytes)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare the Nemotron-CC tokenizer for gpt")
    args = parser.parse_args()

    print(f"Cache directory: {CACHE_DIR}")
    print(f"Data directory: {DATA_DIR}")
    print()

    # Step 1: Verify data is present
    check_data()
    print()

    # Step 2: Train tokenizer
    train_tokenizer()
    print()

    # Step 3: Count total training tokens
    tokenizer = Tokenizer.from_directory()
    train_files = [p for p in list_parquet_files() if not p.endswith(VAL_FILENAME)]
    total_tokens = 0
    for filepath in train_files:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(pf.num_row_groups):
            texts = pf.read_row_group(rg_idx).column('text').to_pylist()
            total_tokens += sum(len(ids) for ids in tokenizer.encode(texts, prepend=None))
    print(f"Total training tokens: {total_tokens:,}  ({total_tokens / 1e9:.3f}B)")
    print()
    print("Done! Ready to train.")
