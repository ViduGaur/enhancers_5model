"""Download (into ./data only) and load genomic-benchmarks datasets as DataFrames."""
import socket

import pandas as pd

from .config import CACHE_DIR, DATA_DIR, POSITIVE_CLASS, VERSION


def use_https():
    """The package fetches metadata from GitHub over plain http://, which hangs on
    some networks. Point it at https:// instead (runtime patch, package files untouched)."""
    import genomic_benchmarks.data_check.info as info_mod
    import genomic_benchmarks.utils.datasets as ds_mod
    import genomic_benchmarks.utils.paths as paths_mod

    url = paths_mod.DATASET_URL_PATH.with_scheme("https")
    for mod in (paths_mod, ds_mod, info_mod):
        mod.DATASET_URL_PATH = url
    socket.setdefaulttimeout(120)


def download(name):
    """Download from the package's Zenodo cache into ./data/<name>. No-op if present."""
    from genomic_benchmarks.loc2seq import download_dataset

    target = DATA_DIR / name
    if (target / "train").is_dir() and (target / "test").is_dir():
        return target
    use_https()
    # dest_path AND cache_path are set so nothing is written to ~/.genomic_benchmarks
    return download_dataset(name, version=VERSION, dest_path=DATA_DIR,
                            cache_path=CACHE_DIR / "fasta", use_cloud_cache=True)


def load(name):
    """Columns: seq (uppercase str), cls (folder name), label (1 = POSITIVE_CLASS), split, file."""
    cache = CACHE_DIR / f"{name}_seqs.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    root = download(name)
    rows = []
    for split in ["train", "test"]:
        for cls_dir in sorted(p for p in (root / split).iterdir() if p.is_dir()):
            for f in sorted(cls_dir.glob("*.txt")):
                raw = f.read_text().strip()
                n_lower = sum(c.islower() for c in raw)  # soft-masked (repeat) bases
                rows.append((raw.upper(), cls_dir.name, split, f.name, n_lower))
    df = pd.DataFrame(rows, columns=["seq", "cls", "split", "file", "n_lower"])
    df["label"] = (df["cls"] == POSITIVE_CLASS).astype(int)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_pickle(cache)
    return df
