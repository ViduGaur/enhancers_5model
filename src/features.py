"""K-mer frequency features with on-disk caching."""
import os
from itertools import product

import numpy as np

_LUT = np.full(256, -1, dtype=np.int8)
for i, b in enumerate("ACGT"):
    _LUT[ord(b)] = i
    _LUT[ord(b.lower())] = i


def kmer_names(k):
    return ["".join(p) for p in product("ACGT", repeat=k)]


def _seq_counts(seq, k, out):
    codes = _LUT[np.frombuffer(seq.encode("ascii"), dtype=np.uint8)].astype(np.int64)
    n = len(codes) - k + 1
    if n <= 0:
        return
    # rolling base-4 index; windows containing a non-ACGT base are dropped
    idx = np.zeros(n, dtype=np.int64)
    bad = np.zeros(n, dtype=bool)
    for j in range(k):
        win = codes[j:j + n]
        idx = idx * 4 + np.maximum(win, 0)
        bad |= win < 0
    out += np.bincount(idx[~bad], minlength=4 ** k)


def kmer_matrix(seqs, k):
    """Rows are k-mer frequencies (counts / total valid k-mers), float32."""
    X = np.zeros((len(seqs), 4 ** k), dtype=np.float32)
    row = np.zeros(4 ** k, dtype=np.float64)
    for i, s in enumerate(seqs):
        row[:] = 0
        _seq_counts(s, k, row)
        tot = row.sum()
        if tot > 0:
            X[i] = row / tot
    return X


def cached_kmer_matrix(seqs, k, cache_path, mmap=False):
    if os.path.exists(cache_path):
        X = np.load(cache_path, mmap_mode="r" if mmap else None)
        if X.shape == (len(seqs), 4 ** k):
            return X
    X = kmer_matrix(seqs, k)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, X)
    return X


def clean_matrices(k):
    """k-mer matrices for cleaned human train/test and mouse windows (cached as .npy)."""
    from .clean import get_clean
    from .config import CACHE_DIR

    human, mouse, _ = get_clean()
    parts = {"human_train": human[human.split == "train"], "human_test": human[human.split == "test"],
             "mouse": mouse}
    out = {}
    for name, df in parts.items():
        X = cached_kmer_matrix(df.seq.tolist(), k, str(CACHE_DIR / "kmers" / f"{name}_k{k}.npy"), mmap=True)
        out[name] = (X, df.label.to_numpy(), df.reset_index(drop=True))
    return out
