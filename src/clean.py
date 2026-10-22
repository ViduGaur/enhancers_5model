"""Data cleaning for the human (train/test) and mouse (evaluation-only) sets.

Every step logs before/after counts per split and class.
"""
import numpy as np
import pandas as pd

from .config import CACHE_DIR, HUMAN, MOUSE
from .data import load

MIN_LEN = 50            # human: drop sequences shorter than this
MOUSE_MAX_N = 0.50      # mouse: drop source sequences with more N than this
WINDOW = 270            # mouse: window length (~ human median 269 bp)
WINDOW_MAX_N = 0.05     # mouse: drop windows with more N than this

_RC = str.maketrans("ACGTN", "TGCAN")


def canonical(s):
    """Strand-independent key: the smaller of a sequence and its reverse complement."""
    rc = s.translate(_RC)[::-1]
    return min(s, rc)


def _counts(step, df, log, by=("split", "cls")):
    for key, n in df.groupby(list(by)).size().items():
        key = key if isinstance(key, tuple) else (key,)
        log.append({"step": step, **dict(zip(by, key)), "n": int(n)})


def clean_human(log):
    df = load(HUMAN)[["seq", "cls", "label", "split"]].copy()
    _counts("human: raw", df, log)

    df = df[df.seq.str.len() >= MIN_LEN]
    _counts(f"human: drop < {MIN_LEN} bp", df, log)

    df = df.assign(key=df.seq.map(canonical))
    # a sequence (either strand) carrying both labels is ambiguous -> drop every copy
    n_lab = df.groupby("key").label.nunique()
    df = df[~df.key.isin(n_lab[n_lab > 1].index)]
    _counts("human: drop label-conflicting seqs", df, log)

    df = df.drop_duplicates(["split", "key"])
    _counts("human: dedupe within split (incl. reverse complement)", df, log)

    train_keys = set(df.loc[df.split == "train", "key"])
    df = df[(df.split == "train") | ~df.key.isin(train_keys)]
    _counts("human: drop test seqs found in train (incl. reverse complement)", df, log)
    return df.reset_index(drop=True)


def clean_mouse(log, seed=0):
    df = load(MOUSE)[["seq", "cls", "label", "split"]].copy()
    df["split"] = "eval"            # mouse is evaluation-only: pool train+test
    df["src"] = np.arange(len(df))  # source sequence id (windows from one source are correlated)
    _counts("mouse: raw (train+test pooled)", df, log)

    df = df[df.seq.str.count("N") / df.seq.str.len() <= MOUSE_MAX_N]
    _counts(f"mouse: drop seqs with > {MOUSE_MAX_N:.0%} N", df, log)

    rows = []
    for r in df.itertuples():
        for start in range(0, len(r.seq) - WINDOW + 1, WINDOW):
            rows.append((r.seq[start:start + WINDOW], r.cls, r.label, r.split, r.src))
    w = pd.DataFrame(rows, columns=["seq", "cls", "label", "split", "src"])
    _counts(f"mouse: cut into {WINDOW} bp non-overlapping windows", w, log)

    w = w[w.seq.str.count("N") / WINDOW <= WINDOW_MAX_N]
    _counts(f"mouse: drop windows with > {WINDOW_MAX_N:.0%} N", w, log)

    w = w.assign(key=w.seq.map(canonical)).drop_duplicates("key")
    _counts("mouse: dedupe windows (incl. reverse complement)", w, log)

    n_min = w.cls.value_counts().min()
    w = w.groupby("cls", group_keys=False).sample(n=n_min, random_state=seed)
    _counts("mouse: subsample majority class to balance", w, log)
    return w.reset_index(drop=True)


def get_clean():
    """Returns (human_df, mouse_df, counts_log_df); cached in .cache/."""
    paths = [CACHE_DIR / f"clean_{n}.pkl" for n in ("human", "mouse", "log")]
    if all(p.exists() for p in paths):
        return tuple(pd.read_pickle(p) for p in paths)
    log = []
    human, mouse = clean_human(log), clean_mouse(log)
    # cross-species leakage check after windowing
    overlap = len(set(human.key) & set(mouse.key))
    log.append({"step": "check: human/mouse identical seqs after cleaning (incl. RC)", "n": overlap})
    log = pd.DataFrame(log)
    for obj, p in zip((human, mouse, log), paths):
        obj.to_pickle(p)
    return human, mouse, log
