"""Clean human + mouse data and append before/after counts to verification_report.md."""
import pandas as pd

from src.clean import MIN_LEN, MOUSE_MAX_N, WINDOW, WINDOW_MAX_N, get_clean
from src.config import RESULTS_DIR, ROOT

human, mouse, log = get_clean()
RESULTS_DIR.mkdir(exist_ok=True)
log.to_csv(RESULTS_DIR / "cleaning_counts.csv", index=False)

counts = log[log.cls.notna()].pivot_table(index="step", columns=["split", "cls"], values="n",
                                          sort=False, aggfunc="sum").fillna(0).astype(int)
counts["total"] = counts.sum(axis=1)
print(counts.to_string())
checks = log[log.cls.isna()]
print(checks.to_string(index=False))


def balance(df):
    vc = df.cls.value_counts()
    return ", ".join(f"{c}={n} ({n / len(df):.1%})" for c, n in vc.items())


h_tr, h_te = human[human.split == "train"], human[human.split == "test"]
lines = [
    "", "## Data cleaning (option A)", "",
    "The checks above ran on the raw data. The human train/test overlap FAIL, the mouse N-content and "
    "length WARNs are handled by the steps below; the mouse size WARN is only mitigated (windows give more, but correlated, test points). All experiments use the cleaned data.", "",
    f"Human: drop sequences < {MIN_LEN} bp; drop every copy of sequences carrying both labels; "
    "dedupe within each split using a strand-independent key (sequence or its reverse complement); "
    "drop test sequences whose key appears in train.",
    f"Mouse (evaluation only, train+test pooled): drop sequences with > {MOUSE_MAX_N:.0%} N; "
    f"cut into non-overlapping {WINDOW} bp windows; drop windows with > {WINDOW_MAX_N:.0%} N; "
    "dedupe windows; subsample the majority class to balance.", "",
    "Counts after each step (split / class):", "", "```", counts.to_string(), "```", "",
    "```", checks[["step", "n"]].to_string(index=False), "```", "",
    f"- Human train after cleaning: {len(h_tr)} ({balance(h_tr)}). Training subsamples are drawn class-balanced.",
    f"- Human test after cleaning: {len(h_te)} ({balance(h_te)}). Imbalanced, so F1 and ROC-AUC are the main "
    "metrics; accuracy is reported on a class-balanced subsample of the test set.",
    f"- Mouse windows after cleaning: {len(mouse)} ({balance(mouse)}) from "
    f"{mouse.src.nunique()} source sequences. Windows from the same source sequence are correlated.",
    f"- Mouse N content after cleaning: {mouse.seq.str.count('N').sum() / mouse.seq.str.len().sum():.3%} of bases; "
    f"by class {(mouse.seq.str.count('N') / mouse.seq.str.len()).groupby(mouse.cls).mean().round(4).to_dict()}.",
    f"- GC by class after cleaning: human train "
    f"{(h_tr.seq.str.count('[GC]') / h_tr.seq.str.len()).groupby(h_tr.cls).mean().round(3).to_dict()}, mouse "
    f"{(mouse.seq.str.count('[GC]') / mouse.seq.str.len()).groupby(mouse.cls).mean().round(3).to_dict()} "
    "(a real compositional difference that models can use).",
]
rep = ROOT / "verification_report.md"
text = rep.read_text().split("\n## Data cleaning")[0].rstrip("\n")
rep.write_text(text + "\n" + "\n".join(lines) + "\n")
print("\n".join(lines[-5:]))
