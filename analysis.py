"""Phase 2 analysis: tables, generalization gap, figures, LogReg k-mer interpretation.

Reads results/runs.csv + results/tuning.csv (works on partial results too).
Best k per model is chosen by human CV AUC only, never by mouse performance.
"""
import json
import re
import urllib.request

import joblib

import src.config  # noqa: F401  (sets MPLCONFIGDIR to ./.cache/mpl before matplotlib loads)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import DATA_DIR, FIGURES_DIR, RESULTS_DIR  # noqa: E402
from src.features import clean_matrices, kmer_names  # noqa: E402
from src.models import MODEL_NAMES  # noqa: E402

FIGURES_DIR.mkdir(exist_ok=True)
# validated categorical palette (fixed order = MODEL_NAMES), plus chart chrome
COLORS = dict(zip(MODEL_NAMES, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
MARKERS = dict(zip(MODEL_NAMES, ["o", "s", "^", "D", "v"]))
HUMAN_C, MOUSE_C = "#2a78d6", "#eb6834"
SHORT = {"LinearSVM": "Linear\nSVM", "RandomForest": "Random\nForest"}
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "legend.frameon": False, "lines.linewidth": 2,
})

METRICS = ["human_f1", "human_auc", "human_bal_acc", "mouse_acc", "mouse_f1", "mouse_auc", "mouse_src_auc"]


def to_md(df, floatfmt="{:.3f}"):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(out)


def pm(mean, std):
    return f"{mean:.3f} ± {std:.3f}"


runs = pd.read_csv(RESULTS_DIR / "runs.csv")
tuning = pd.read_csv(RESULTS_DIR / "tuning.csv")
models = runs[runs.kind == "model"].copy()
models["gap_auc"] = models.human_auc - models.mouse_auc
models["gap_acc"] = models.human_bal_acc - models.mouse_acc
ks = sorted(models.k.unique())
names = [m for m in MODEL_NAMES if m in set(models.model)]

# ---------------------------------------------------------------- results table (mean ± std over seeds)
agg = models.groupby(["model", "k"])[METRICS + ["gap_auc", "gap_acc"]].agg(["mean", "std"])
agg = agg.reindex([(m, k) for m in names for k in ks if (m, k) in agg.index])
flat = agg.copy()
flat.columns = [f"{a}_{b}" for a, b in flat.columns]
flat = flat.reset_index().merge(tuning[["model", "k", "cv_auc_mean", "best_params"]], on=["model", "k"], how="left")
flat.to_csv(RESULTS_DIR / "results_table.csv", index=False)

best_k = tuning.loc[tuning.groupby("model").cv_auc_mean.idxmax(), ["model", "k", "cv_auc_mean", "best_params"]]
best_k = best_k.set_index("model").reindex(names)

table = pd.DataFrame({
    "model": [m for m, _ in agg.index], "k": [k for _, k in agg.index],
    "CV AUC": flat.cv_auc_mean.round(3).values,
    **{lab: [pm(agg.loc[i, (c, "mean")], agg.loc[i, (c, "std")]) for i in agg.index]
       for lab, c in [("human F1", "human_f1"), ("human AUC", "human_auc"), ("human bal. acc", "human_bal_acc"),
                      ("mouse acc", "mouse_acc"), ("mouse F1", "mouse_f1"), ("mouse AUC", "mouse_auc"),
                      ("AUC gap", "gap_auc")]},
})
table["best k (CV)"] = ["*" if best_k.loc[m, "k"] == k else "" for m, k in zip(table.model, table.k)]

controls = runs[runs.kind.isin(["shuffled", "baseline"])].copy()
controls["control"] = np.where(controls.kind == "baseline", "majority class", "shuffled labels")
ctrl = (controls.groupby(["control", "model"])[["human_acc", "human_bal_acc", "human_f1", "human_auc",
                                                 "mouse_acc", "mouse_auc"]].agg(["mean", "max"]))
ctrl_tbl = pd.DataFrame({
    "control": [c for c, _ in ctrl.index], "model": [m for _, m in ctrl.index],
    **{f"{c} (mean / max over k)": [f"{ctrl.loc[i, (c, 'mean')]:.3f} / {ctrl.loc[i, (c, 'max')]:.3f}"
                                    for i in ctrl.index]
       for c in ["human_acc", "human_bal_acc", "human_f1", "human_auc", "mouse_acc", "mouse_auc"]},
})
controls.to_csv(RESULTS_DIR / "controls.csv", index=False)

perm = runs[runs.kind == "perm_null"]
perm_tbl = pd.DataFrame()
if len(perm):
    g = perm.groupby("k")
    perm_tbl = pd.DataFrame({
        "k": list(g.groups),
        "n perms": g.size().values,
        **{lab: [f"{d[c].mean():.3f} ± {d[c].std():.3f} [{d[c].min():.3f}, {d[c].max():.3f}]" for _, d in g]
           for lab, c in [("human AUC", "human_auc"), ("human bal. acc", "human_bal_acc"),
                          ("mouse acc", "mouse_acc"), ("mouse AUC", "mouse_auc")]},
    })
    perm_tbl.to_csv(RESULTS_DIR / "permutation_null.csv", index=False)

# ---------------------------------------------------------------- generalization gap ranking
rank = []
for m in names:
    kb = int(best_k.loc[m, "k"])
    sel = models[(models.model == m) & (models.k == kb)]
    allk = models[models.model == m].groupby("k").gap_auc.mean()
    rank.append({"model": m, "best k (CV)": kb,
                 "human AUC": pm(sel.human_auc.mean(), sel.human_auc.std()),
                 "mouse AUC": pm(sel.mouse_auc.mean(), sel.mouse_auc.std()),
                 "AUC gap @best k": sel.gap_auc.mean(), "AUC gap std": sel.gap_auc.std(),
                 "acc gap @best k (bal. human - mouse)": sel.gap_acc.mean(),
                 "mean AUC gap over k": allk.mean()})
rank = pd.DataFrame(rank).sort_values("AUC gap @best k").reset_index(drop=True)
rank.insert(0, "rank", np.arange(1, len(rank) + 1))
rank.to_csv(RESULTS_DIR / "generalization_gap.csv", index=False)
gap_k = models.groupby(["model", "k"])[["gap_auc", "gap_acc"]].agg(["mean", "std"])
gap_k.columns = [f"{a}_{b}" for a, b in gap_k.columns]
gap_k.reset_index().to_csv(RESULTS_DIR / "gap_by_k.csv", index=False)

md = ["# Results", "",
      "All numbers: mean ± std over 3 seeds (each seed = new class-balanced 20k human training subsample).",
      "Human test is the cleaned (deduplicated, leakage-free) set, 12.6% positives: F1 and AUC are the main metrics; "
      "'bal. acc' = accuracy on a class-balanced subsample of it. Mouse = 270 bp windows, balanced, evaluation only.",
      "AUC gap = human test AUC - mouse AUC (positive = performance drops on mouse). "
      "'best k (CV)' = k with the highest human 5-fold CV AUC.", "",
      "## Main table", "", to_md(table), "",
      "## Generalization gap ranking (smallest drop first, at each model's best k)", "", to_md(rank), "",
      "## Sanity controls", "", to_md(ctrl_tbl), "",
      "## Permutation null (LogReg, 10 label shuffles per k; mean ± std [min, max])", "",
      to_md(perm_tbl) if len(perm_tbl) else "(not run yet)", ""]
(RESULTS_DIR / "results_table.md").write_text("\n".join(md))
print("\n".join(md))

# ---------------------------------------------------------------- figure 1: human vs mouse per model (best k)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
x = np.arange(len(names))
w = 0.38
for ax, (hc, mc, title, ylab) in zip(axes, [
        ("human_auc", "mouse_auc", "ROC-AUC", "ROC-AUC"),
        ("human_bal_acc", "mouse_acc", "Accuracy (balanced eval sets)", "accuracy")]):
    for j, (col, color, lab) in enumerate([(hc, HUMAN_C, "human test"), (mc, MOUSE_C, "mouse (cross-species)")]):
        vals, errs = [], []
        for m in names:
            sel = models[(models.model == m) & (models.k == best_k.loc[m, "k"])][col]
            vals.append(sel.mean())
            errs.append(sel.std())
        bars = ax.bar(x + (j - 0.5) * (w + 0.02), vals, w, yerr=errs, color=color, label=lab,
                      error_kw={"ecolor": INK2, "elinewidth": 1, "capsize": 3}, edgecolor="#fcfcfb", linewidth=1)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=8, color=INK2)
    ax.axhline(0.5, color=MUTED, lw=1, ls="--")
    ax.text(len(names) - 0.45, 0.5, "chance", color=MUTED, fontsize=8, ha="left", va="center", clip_on=False)
    ax.set_xticks(x, [f"{SHORT.get(m, m)}\nk={int(best_k.loc[m, 'k'])}" for m in names])
    ax.set_ylim(0.4, 0.9)
    ax.set_ylabel(ylab)
    ax.set_title(title, loc="left", fontsize=11)
    ax.grid(axis="x", visible=False)
axes[0].legend(loc="upper left", ncols=2)
fig.suptitle("Human test vs. mouse performance per model (best k by human CV; mean ± std, 3 seeds)",
             x=0.01, ha="left", fontsize=12)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "human_vs_mouse.png", dpi=150)
plt.close(fig)


# ---------------------------------------------------------------- figure 2: gap across k, figure 3: AUC vs k
def line_panel(ax, col, ylabel, title, zero=None):
    for m in names:
        g = models[models.model == m].groupby("k")[col].agg(["mean", "std"])
        ax.errorbar(g.index, g["mean"], yerr=g["std"], color=COLORS[m], marker=MARKERS[m], ms=7,
                    capsize=3, label=m, mec="#fcfcfb", mew=1.5)
    if zero is not None:
        ax.axhline(zero, color=MUTED, lw=1, ls="--")
    ax.set_xticks(ks)
    ax.set_xlabel("k-mer length k")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=11)


fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
line_panel(axes[0], "gap_auc", "human AUC - mouse AUC", "AUC gap (lower = generalizes better)", zero=0)
line_panel(axes[1], "gap_acc", "human bal. acc - mouse acc", "Accuracy gap", zero=0)
axes[1].legend(loc="best")
fig.suptitle("Cross-species generalization gap across k (mean ± std, 3 seeds)", x=0.01, ha="left", fontsize=12)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "gap_vs_k.png", dpi=150)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
line_panel(axes[0], "human_auc", "ROC-AUC", "Human test AUC")
line_panel(axes[1], "mouse_auc", "", "Mouse AUC (cross-species)")
axes[0].legend(loc="lower right")
fig.suptitle("Effect of k on performance (mean ± std, 3 seeds)", x=0.01, ha="left", fontsize=12)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "auc_vs_k.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------- figure 4: confusion matrices (best k, seed 0)
fig, axes = plt.subplots(2, len(names), figsize=(2.6 * len(names), 5.4))
axes = np.atleast_2d(axes)
for j, m in enumerate(names):
    r = models[(models.model == m) & (models.k == best_k.loc[m, "k"]) & (models.seed == 0)].iloc[0]
    for i, (col, lab) in enumerate([("human_cm", "human test"), ("mouse_cm", "mouse")]):
        cm = np.array(json.loads(r[col])).reshape(2, 2)
        norm = cm / cm.sum(1, keepdims=True)
        ax = axes[i, j]
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for a in range(2):
            for b in range(2):
                ax.text(b, a, f"{cm[a, b]}\n{norm[a, b]:.0%}", ha="center", va="center", fontsize=8,
                        color="#ffffff" if norm[a, b] > 0.6 else INK)
        ax.set_xticks([0, 1], ["pred bg", "pred enh"], fontsize=8)
        ax.set_yticks([0, 1], ["true bg", "true enh"], fontsize=8)
        ax.grid(False)
        ax.set_title(f"{m} k={int(best_k.loc[m, 'k'])}\n{lab}", fontsize=9)
fig.suptitle("Confusion matrices (seed 0; row-normalized %)", x=0.01, ha="left", fontsize=12)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "confusion_matrices.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------- interpretability: LogReg top k-mers
k_lr = int(best_k.loc["LogReg", "k"])
pipe = joblib.load(RESULTS_DIR / "models" / f"LogReg_k{k_lr}_seed0.joblib")
coef = pipe[-1].coef_[0]  # weights on standardized features (comparable across k-mers)
kn = kmer_names(k_lr)
Xtr, ytr, _ = clean_matrices(k_lr)["human_train"]
f_pos = np.asarray(Xtr[ytr == 1].mean(0)).ravel()
f_neg = np.asarray(Xtr[ytr == 0].mean(0)).ravel()
rc = str.maketrans("ACGT", "TGCA")
kdf = pd.DataFrame({"kmer": kn, "reverse_complement": [s.translate(rc)[::-1] for s in kn], "weight": coef,
                    "freq_enhancer": f_pos, "freq_background": f_neg,
                    "gc": [(s.count("G") + s.count("C")) / k_lr for s in kn]})
top_pos = kdf.nlargest(20, "weight").assign(direction="enhancer (+)")
top_neg = kdf.nsmallest(20, "weight").assign(direction="background (-)")
top = pd.concat([top_pos, top_neg]).reset_index(drop=True)


# ---- JASPAR comparison (only if network works): heuristic best-match of each k-mer to CORE vertebrate PFMs
def load_jaspar():
    path = DATA_DIR / "jaspar" / "JASPAR_CORE_vertebrates_nonredundant_pfms.txt"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        for year in (2024, 2022):
            url = (f"https://jaspar.elixir.no/download/data/{year}/CORE/"
                   f"JASPAR{year}_CORE_vertebrates_non-redundant_pfms_jaspar.txt")
            try:
                with urllib.request.urlopen(url, timeout=60) as r:
                    path.write_bytes(r.read())
                break
            except Exception as e:  # noqa: BLE001
                print(f"JASPAR download failed ({url}): {e}")
        else:
            return None
    motifs, cur = [], None
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            cur = {"id": line[1:].split()[0], "name": " ".join(line[1:].split()[1:]), "rows": {}}
            motifs.append(cur)
        elif cur is not None and line.strip():
            base = line.strip()[0]
            nums = [float(v) for v in re.findall(r"[-\d.]+", line.strip()[1:])]
            cur["rows"][base] = nums
    out = []
    for mo in motifs:
        if set(mo["rows"]) != set("ACGT"):
            continue
        m = np.array([mo["rows"][b] for b in "ACGT"]) + 0.8  # pseudocounts
        p = m / m.sum(0)
        out.append((mo["id"], mo["name"], np.log2(p / 0.25)))
    return out


def best_match(kmer, motifs, min_core=1.0, hit=0.95):
    """Match the k-mer (either strand) only against motif *core* windows, where every position's
    consensus base has log2-odds >= min_core (probability >= 0.5). Returns the best match by
    (relative score, absolute log-odds) and the number of motifs with a relative score >= hit."""
    idx = {b: i for i, b in enumerate("ACGT")}
    best, hits = (0.0, -np.inf, "", ""), set()
    for seq in {kmer, kmer.translate(rc)[::-1]}:
        s = np.array([idx[c] for c in seq])
        for mid, name, lo in motifs:
            for st in range(lo.shape[1] - len(seq) + 1):
                win = lo[:, st:st + len(seq)]
                if win.max(0).min() < min_core:
                    continue
                score = win[s, np.arange(len(seq))].sum()
                mx, mn = win.max(0).sum(), win.min(0).sum()
                rel = (score - mn) / (mx - mn)
                if rel >= hit:
                    hits.add(mid)
                if (rel, score) > best[:2]:
                    best = (rel, score, mid, name)
    return best[0], best[2], best[3], len(hits)


jaspar_note = ""
motifs = load_jaspar() if k_lr >= 5 else None
if motifs:
    matches = [best_match(km, motifs) for km in top.kmer]
    top["jaspar_best_id"] = [m[1] for m in matches]
    top["jaspar_best_name"] = [m[2] for m in matches]
    top["jaspar_rel_score"] = [round(m[0], 3) for m in matches]
    top["jaspar_n_motifs_hit"] = [m[3] for m in matches]
    jaspar_note = (f"JASPAR CORE vertebrates (non-redundant), {len(motifs)} PFMs. Each k-mer (either strand) is "
                   "scored only against motif core windows (consensus base probability >= 0.5 at every position); "
                   "jaspar_rel_score is the best relative log-odds score (1 = matches the core consensus) and "
                   "jaspar_n_motifs_hit counts motifs with score >= 0.95. Heuristic: a 5-mer that hits many "
                   "motifs is not specific; treat matches as hints for manual checking, not as evidence.")
elif k_lr < 5:
    jaspar_note = f"Best LogReg k is {k_lr}: too short for meaningful motif matching; JASPAR comparison is manual."
else:
    jaspar_note = "JASPAR could not be downloaded; comparison is manual (https://jaspar.elixir.no)."
print(jaspar_note)
top.to_csv(RESULTS_DIR / "logreg_top_kmers.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(11, 6.5))
for ax, (d, color, title) in zip(axes, [(top_pos, HUMAN_C, "Top 20 enhancer-associated k-mers (+)"),
                                        (top_neg, MOUSE_C, "Top 20 background-associated k-mers (-)")]):
    d = d.iloc[::-1]
    ax.barh(d.kmer, d.weight, color=color, height=0.7)
    ax.axvline(0, color=AXIS, lw=1)
    ax.set_title(title, loc="left", fontsize=11)
    ax.set_xlabel("LogReg weight (standardized feature)")
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", visible=False)
    for lab in ax.get_yticklabels():
        lab.set_fontfamily("monospace")
        lab.set_color(INK2)
fig.suptitle(f"Logistic regression k-mer weights (k={k_lr}, seed 0)", x=0.01, ha="left", fontsize=12)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "logreg_top_kmers.png", dpi=150)
plt.close(fig)

summary = {"best_k": best_k.k.astype(int).to_dict(), "logreg_k": k_lr, "jaspar_note": jaspar_note,
           "gap_ranking": rank[["model", "AUC gap @best k"]].values.tolist()}
(RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
print(top[["direction", "kmer", "reverse_complement", "weight", "gc"]
          + [c for c in top.columns if c.startswith("jaspar")]].to_string(index=False))
