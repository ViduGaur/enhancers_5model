"""Phase 1: verify environment, data, featurizer and model feasibility.

Prints a PASS/WARN/FAIL report and writes verification_report.md.
Exit code 1 if any check FAILs.
"""
import sys
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.config import HUMAN, MOUSE, ROOT
from src.data import load
from src.features import kmer_matrix
from src.models import make_model, MODEL_NAMES, GRIDS

RNG = np.random.default_rng(0)
checks = []      # (section, item, status, detail)
details = []     # extra markdown blocks


def check(section, item, status, detail=""):
    checks.append((section, item, status, detail))
    print(f"[{status:4}] {section} | {item}: {detail}")


def md_table(df):
    return df.to_markdown() if hasattr(df, "to_markdown") and _has_tabulate() else "```\n" + df.to_string() + "\n```"


def _has_tabulate():
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


def revcomp(s):
    return s[::-1].translate(str.maketrans("ACGTN", "TGCAN"))


# ---------------------------------------------------------------- environment
venv = (ROOT / ".venv").resolve()
check("env", "python", "PASS", sys.version.split()[0])
check("env", "project-local venv",
      "PASS" if Path(sys.prefix).resolve() == venv else "FAIL", sys.prefix)
pkgs = ["numpy", "pandas", "scikit-learn", "matplotlib", "seaborn", "genomic-benchmarks", "tqdm", "joblib"]
check("env", "packages", "PASS", ", ".join(f"{p}={version(p)}" for p in pkgs))

# ---------------------------------------------------------------- datasets
from genomic_benchmarks.data_check import list_datasets  # noqa: E402

available = sorted(list_datasets())
check("data", "list_datasets()", "PASS", ", ".join(available))
mouse_like = [d for d in available if "mouse" in d]
check("data", "mouse datasets in package",
      "PASS" if mouse_like else "FAIL", ", ".join(mouse_like) or "none")
for name in [HUMAN, MOUSE]:
    check("data", f"{name} listed", "PASS" if name in available else "FAIL")

dfs = {}
for name in [HUMAN, MOUSE]:
    t = time.time()
    try:
        dfs[name] = load(name)
        ok = (config.DATA_DIR / name).is_dir()
        check("data", f"{name} download+load", "PASS" if ok else "FAIL",
              f"{len(dfs[name])} seqs in {time.time() - t:.0f}s -> data/{name}/")
    except Exception as e:  # noqa: BLE001
        check("data", f"{name} download+load", "FAIL", f"{type(e).__name__}: {e}")
if len(dfs) < 2:
    print("Cannot continue without both datasets.")
    sys.exit(1)

home_cache = Path.home() / ".genomic_benchmarks"
check("data", "nothing written to ~/.genomic_benchmarks",
      "PASS" if not home_cache.exists() else "WARN",
      "absent" if not home_cache.exists() else f"{home_cache} exists (check if pre-existing)")

# ---------------------------------------------------------------- per-dataset stats
stat_rows = []
for name, df in dfs.items():
    df = df.copy()
    df["len"] = df.seq.str.len()
    df["non_acgt"] = df.seq.str.count(r"[^ACGT]")
    df["gc"] = df.seq.str.count(r"[GC]") / df["len"]
    classes = sorted(df.cls.unique())
    check(name, "classes", "PASS" if len(classes) == 2 and config.POSITIVE_CLASS in classes else "FAIL",
          f"{classes} (label 1 = '{config.POSITIVE_CLASS}')")

    for split in ["train", "test"]:
        d = df[df.split == split]
        counts = d.cls.value_counts().to_dict()
        minority = min(counts.values()) / len(d) if len(d) else 0
        status = "PASS" if minority >= 0.4 else ("WARN" if minority >= 0.2 else "FAIL")
        check(name, f"{split} size/balance", status,
              f"n={len(d)} {counts} minority={minority:.1%}")

    for cls in classes:
        d = df[df.cls == cls]
        stat_rows.append({
            "dataset": name, "class": cls, "n": len(d),
            "len_min": d.len.min(), "len_max": d.len.max(), "len_mean": round(d.len.mean(), 1),
            "gc_mean": round(d.gc.mean(), 3),
            "seqs_with_non_ACGT": int((d.non_acgt > 0).sum()),
            "non_ACGT_bases": int(d.non_acgt.sum()),
            "softmasked_frac": round(d.n_lower.sum() / d.len.sum(), 3),
        })

    total_len = df.len.sum()
    bad = df.non_acgt.sum()
    frac_seqs = (df.non_acgt > 0).mean()
    other_chars = sorted(set("".join(df.seq)) - set("ACGT"))
    check(name, "alphabet", "PASS" if frac_seqs < 0.01 else "WARN",
          f"non-ACGT chars {other_chars or 'none'}; {bad} bases ({bad / total_len:.4%}); "
          f"{frac_seqs:.2%} of seqs affected")
    check(name, "length", "PASS",
          f"min={df.len.min()} max={df.len.max()} mean={df.len.mean():.1f} median={df.len.median():.0f}")

    # length/GC differences between classes can act as shortcuts; report, don't fail
    by = df.groupby("cls")[["len", "gc"]].mean()
    gc_gap = abs(by.gc.iloc[0] - by.gc.iloc[1])
    check(name, "class GC gap", "PASS" if gc_gap < 0.05 else "WARN",
          f"{gc_gap:.3f} ({by.gc.round(3).to_dict()})")
    len_ratio = by.len.max() / by.len.min()
    check(name, "class length ratio", "PASS" if len_ratio < 1.2 else "WARN",
          f"{len_ratio:.2f} ({by.len.round(1).to_dict()})")

    # duplicates
    for split in ["train", "test"]:
        s = df[df.split == split]
        dup = s.seq.duplicated().sum()
        conflict = (s.groupby("seq").label.nunique() > 1).sum()
        check(name, f"duplicates within {split}", "PASS" if dup / len(s) < 0.01 and conflict == 0 else "WARN",
              f"{dup} duplicate seqs ({dup / len(s):.2%}), {conflict} with conflicting labels")
    tr, te = set(df[df.split == "train"].seq), set(df[df.split == "test"].seq)
    te_rc = {revcomp(s) for s in te}
    cross, cross_rc = len(tr & te), len(tr & te_rc)
    status = "PASS" if cross == 0 and cross_rc == 0 else ("WARN" if (cross + cross_rc) / len(te) < 0.01 else "FAIL")
    check(name, "train/test overlap", status,
          f"{cross} identical, {cross_rc} reverse-complement matches (of {len(te)} test seqs)")

    # label sanity examples
    lines = [f"### {name}: example sequences (first 60 bp)"]
    for cls in classes:
        for _, r in df[df.cls == cls].sample(3, random_state=0).iterrows():
            lines.append(f"- `{cls}` ({r.split}, len {r.len}, GC {r.gc:.2f}): `{r.seq[:60]}...`")
    details.append("\n".join(lines))
    print("\n".join(lines))

h_all, m_all = set(dfs[HUMAN].seq), set(dfs[MOUSE].seq)
m_rc = {revcomp(s) for s in m_all}
ov, ov_rc = len(h_all & m_all), len(h_all & m_rc)
check("cross-species", "human/mouse sequence overlap", "PASS" if ov + ov_rc == 0 else "FAIL",
      f"{ov} identical, {ov_rc} reverse-complement")
h_len, m_len = dfs[HUMAN].seq.str.len(), dfs[MOUSE].seq.str.len()
ratio = m_len.mean() / h_len.mean()
check("cross-species", "length distribution match", "PASS" if 0.67 < ratio < 1.5 else "WARN",
      f"human mean {h_len.mean():.0f} bp (median {h_len.median():.0f}), mouse mean {m_len.mean():.0f} bp "
      f"(median {m_len.median():.0f}); ratio {ratio:.1f}x")
for name, lens in [(HUMAN, h_len), (MOUSE, m_len)]:
    short = {k: int((lens < k + 20).sum()) for k in config.K_VALUES}
    check(name, "very short seqs (< k+20 bp)", "PASS" if max(short.values()) / len(lens) < 0.001 else "WARN",
          f"{short}; shortest {lens.min()} bp")
n_mouse = len(dfs[MOUSE])
check("cross-species", "mouse set size", "PASS" if n_mouse >= 5000 else ("WARN" if n_mouse >= 1000 else "FAIL"),
      f"{n_mouse} sequences total")

stats = pd.DataFrame(stat_rows)
print(stats.to_string(index=False))

# ---------------------------------------------------------------- featurizer test
seqs100 = dfs[HUMAN].seq.iloc[RNG.choice(len(dfs[HUMAN]), 100, replace=False)].tolist()
for k in config.K_VALUES:
    X = kmer_matrix(seqs100, k)
    rs = X.sum(1)
    ok = X.shape == (100, 4 ** k) and np.allclose(rs, 1, atol=1e-4) and not np.isnan(X).any()
    check("features", f"k={k}", "PASS" if ok else "FAIL",
          f"shape={X.shape} row sums [{rs.min():.5f}, {rs.max():.5f}] NaN={int(np.isnan(X).sum())}")
# hand-checked toy example
Xt = kmer_matrix(["ACGTA"], 2)[0]
names2 = ["".join(p) for p in __import__("itertools").product("ACGT", repeat=2)]
expect = {"AC": .25, "CG": .25, "GT": .25, "TA": .25}
ok = all(abs(Xt[names2.index(km)] - v) < 1e-6 for km, v in expect.items()) and abs(Xt.sum() - 1) < 1e-6
check("features", "toy 'ACGTA' k=2", "PASS" if ok else "FAIL", "AC,CG,GT,TA = 0.25 each")

# ---------------------------------------------------------------- smoke test
human, mouse = dfs[HUMAN], dfs[MOUSE]
h_tr, h_te = human[human.split == "train"], human[human.split == "test"]


def sample(df, n):
    return df.iloc[RNG.choice(len(df), min(n, len(df)), replace=False)]


tr500, te500, mo500 = sample(h_tr, 500), sample(h_te, 500), sample(mouse, 500)
Xtr, Xte, Xmo = (kmer_matrix(d.seq.tolist(), 3) for d in (tr500, te500, mo500))
smoke = []
for name in MODEL_NAMES:
    m = make_model(name, seed=0)
    t = time.time()
    m.fit(Xtr, tr500.label.values)
    fit_t = time.time() - t
    smoke.append({"model": name, "fit_s": round(fit_t, 3),
                  "human_test_acc": round((m.predict(Xte) == te500.label.values).mean(), 3),
                  "mouse_acc": round((m.predict(Xmo) == mo500.label.values).mean(), 3)})
    check("smoke k=3 n=500", name, "PASS", f"fit {fit_t:.2f}s, human acc {smoke[-1]['human_test_acc']}, "
          f"mouse acc {smoke[-1]['mouse_acc']}")
from sklearn.svm import SVC  # noqa: E402
for n in [500, 2000]:
    d = sample(h_tr, n)
    t = time.time()
    SVC(kernel="rbf").fit(kmer_matrix(d.seq.tolist(), 6), d.label.values)
    check("smoke", f"RBF SVC k=6 n={n}", "PASS", f"fit {time.time() - t:.2f}s (for LinearSVC vs RBF decision)")
smoke = pd.DataFrame(smoke)

# ---------------------------------------------------------------- runtime estimate
N_tr = len(h_tr)
N_eval = len(h_te) + len(mouse)
t = time.time()
kmer_matrix(sample(human, 2000).seq.tolist(), 6)
feat_per_seq = (time.time() - t) / 2000
feat_total = feat_per_seq * (len(human) + len(mouse)) * len(config.K_VALUES)

# time each model at k=6 for two sizes and extrapolate with the fitted exponent
n1, n2 = 1000, 4000
X1d, X2d, Xev = sample(h_tr, n1), sample(h_tr, n2), sample(h_te, 1000)
X1, X2, Xe = (kmer_matrix(d.seq.tolist(), 6) for d in (X1d, X2d, Xev))
dsum = sum(4 ** k for k in config.K_VALUES) / 4 ** 6   # feature-size factor across k (cost ~ d)


def est_total(N):
    rows = []
    for name in MODEL_NAMES:
        ts = []
        for X, d in [(X1, X1d), (X2, X2d)]:
            m = make_model(name, seed=0, n_jobs=-1)
            t0 = time.time()
            m.fit(X, d.label.values)
            m.predict(Xe)
            ts.append(time.time() - t0)
        b = float(np.clip(np.log(max(ts[1], 1e-3) / max(ts[0], 1e-3)) / np.log(n2 / n1), 1.0, 2.0))
        per_fit = lambda n: ts[1] * (n / n2) ** b  # noqa: E731
        if name == "KNN":  # cost ~ n_train * n_eval
            per_fit = lambda n, t2=ts[1]: t2 * (n / n2) * (max(n * 0.25, 1000) / 1000)  # noqa: E731
        n_grid = int(np.prod([len(v) for v in GRIDS[name].values()]))
        n_seeds = len(config.SEEDS)  # every model is refit per seed (new subsample)
        cv = 5 * n_grid * per_fit(0.8 * N)
        final = (n_seeds + 1) * per_fit(N)  # +1 = shuffled-label control
        rows.append({"model": name, "N_train": N, "t_n1000": round(ts[0], 2), "t_n4000": round(ts[1], 2),
                     "exponent": round(b, 2), "est_minutes_all_k": round((cv + final) * dsum / 60, 1)})
    return rows


est = pd.DataFrame(est_total(N_tr))
est_sub = est.copy()
# alternative: every fit (CV and final) on a 20k-sequence training subsample
for i, r in est_sub.iterrows():
    scale = 20000 / N_tr
    est_sub.loc[i, "N_train"] = 20000
    est_sub.loc[i, "est_minutes_all_k"] = round(r.est_minutes_all_k * scale ** r.exponent, 1)
mem_gb = N_tr * 4 ** 6 * 4 / 1e9
check("runtime", "featurization (all k, all seqs)", "PASS", f"~{feat_total / 60:.1f} min")
check("runtime", "k=6 train matrix memory", "PASS" if mem_gb < 4 else "WARN", f"{mem_gb:.2f} GB float32")
full_total = est.est_minutes_all_k.sum()
sub_total = est_sub.est_minutes_all_k.sum()
check("runtime", f"full experiment (N_train={N_tr})", "PASS" if full_total < 60 else "WARN",
      f"~{full_total:.0f} min (rough; 12 cores)")
check("runtime", "subsampled experiment (N_train=20000)", "PASS" if sub_total < 60 else "WARN",
      f"~{sub_total:.0f} min (rough; 12 cores)")
print(est.to_string(index=False))
print(est_sub.to_string(index=False))

# ---------------------------------------------------------------- report
n_fail = sum(c[2] == "FAIL" for c in checks)
n_warn = sum(c[2] == "WARN" for c in checks)
rep = ["# Verification report", "",
       f"Human: `{HUMAN}`  |  Mouse: `{MOUSE}`  |  {len(checks)} checks: "
       f"{len(checks) - n_fail - n_warn} PASS, {n_warn} WARN, {n_fail} FAIL", "",
       "| section | check | status | detail |", "|---|---|---|---|"]
rep += [f"| {s} | {i} | **{st}** | {d.replace('|', '/')} |" for s, i, st, d in checks]
rep += ["", "## Per-class statistics", "", md_table(stats.set_index(["dataset", "class"])), ""]
rep += details
rep += ["", "## Smoke test (k=3, 500 train seqs, 500 eval seqs)", "", md_table(smoke.set_index("model")),
        "", "## Runtime estimate (k=6 timings extrapolated, summed over k=3..6)", "",
        "Full training set:", "", md_table(est.set_index("model")), "",
        "CV on a 20k subsample:", "", md_table(est_sub.set_index("model")), "",
        "Estimates are rough (power-law fit from two sizes); treat them as order-of-magnitude."]
(ROOT / "verification_report.md").write_text("\n".join(rep) + "\n")

print(f"\nSUMMARY: {len(checks) - n_fail - n_warn} PASS, {n_warn} WARN, {n_fail} FAIL "
      f"-> verification_report.md")
for c in checks:
    if c[2] != "PASS":
        print(f"  {c[2]}: {c[0]} | {c[1]}: {c[3]}")
sys.exit(1 if n_fail else 0)
