"""Phase 2: tune on human (CV), test on human, test cross-species on mouse.

Results are appended to results/*.csv as each (k, model) finishes, and a rerun
skips anything already done, so an interrupted run loses at most one block.
Mouse data is only ever passed to predict(); it is never used for tuning.
"""
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.config import K_VALUES, RESULTS_DIR, SEEDS
from src.features import clean_matrices
from src.models import CV_JOBS, GRIDS, MODEL_NAMES, make_model, scores

N_TRAIN = 20000   # class-balanced training subsample per seed (all models)
N_FOLDS = 5
RUNS = RESULTS_DIR / "runs.csv"
TUNING = RESULTS_DIR / "tuning.csv"
MODELS_DIR = RESULTS_DIR / "models"


def balanced_idx(y, n, seed):
    rng = np.random.default_rng(seed)
    per = min(n // 2, *(np.sum(y == c) for c in (0, 1)))
    return np.sort(np.concatenate([rng.choice(np.flatnonzero(y == c), per, replace=False) for c in (0, 1)]))


def metrics(y, pred, score, prefix):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {f"{prefix}_acc": accuracy_score(y, pred), f"{prefix}_f1": f1_score(y, pred, zero_division=0),
            f"{prefix}_auc": roc_auc_score(y, score) if score is not None else 0.5,
            f"{prefix}_cm": json.dumps([int(tn), int(fp), int(fn), int(tp)])}


def evaluate_all(pred_fn, M, test_bal):
    Xh, yh, _ = M["human_test"]
    Xm, ym, mdf = M["mouse"]
    ph, sh = pred_fn(Xh)
    pm, sm = pred_fn(Xm)
    out = metrics(yh, ph, sh, "human")
    out["human_bal_acc"] = accuracy_score(yh[test_bal], ph[test_bal])
    out.update(metrics(ym, pm, sm, "mouse"))
    if sm is not None:  # sequence-level mouse AUC: average window scores per source sequence
        g = pd.DataFrame({"src": mdf.src, "y": ym, "s": sm}).groupby("src").agg(y=("y", "first"), s=("s", "mean"))
        out["mouse_src_auc"] = roc_auc_score(g.y, g.s)
    else:
        out["mouse_src_auc"] = 0.5
    return out


def append(path, rows):
    pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)


def done_keys(path, cols):
    if not path.exists():
        return set()
    return set(map(tuple, pd.read_csv(path)[cols].astype(str).values))


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    t_all = time.time()
    for k in K_VALUES:
        t = time.time()
        M = clean_matrices(k)
        X, y, _ = M["human_train"]
        yh = M["human_test"][1]
        # fixed class-balanced subsample of the human test set for "balanced accuracy"
        test_bal = balanced_idx(yh, 2 * int(min(np.bincount(yh))), seed=0)
        print(f"\n=== k={k}: features ready in {time.time() - t:.0f}s, train {X.shape}", flush=True)

        runs_done = done_keys(RUNS, ["k", "model", "kind", "seed"])
        tune_done = pd.read_csv(TUNING) if TUNING.exists() else pd.DataFrame(columns=["k", "model"])

        if (str(k), "Majority", "baseline", "0") not in runs_done:
            maj = int(np.bincount(y).argmax())  # majority class of the full cleaned human train set
            row = {"k": k, "model": "Majority", "kind": "baseline", "seed": 0, "params": "{}"}
            row.update(evaluate_all(lambda Z: (np.full(len(Z), maj), None), M, test_bal))
            row["seconds"] = 0.0  # keep the same columns as model rows
            append(RUNS, [row])

        for name in MODEL_NAMES:
            # ---- tuning: 5-fold stratified CV on the seed-0 human subsample (ROC-AUC)
            prev = tune_done[(tune_done.k == k) & (tune_done.model == name)]
            if len(prev):
                best = json.loads(prev.iloc[0].best_params)
            else:
                t = time.time()
                idx = balanced_idx(y, N_TRAIN, seed=0)
                jobs = CV_JOBS[name]
                gs = GridSearchCV(make_model(name, seed=0, n_jobs=1 if jobs > 1 else -1), GRIDS[name],
                                  scoring="roc_auc", cv=StratifiedKFold(N_FOLDS, shuffle=True, random_state=0),
                                  n_jobs=jobs, refit=False)
                gs.fit(X[idx], y[idx])
                best = gs.best_params_
                cvr = pd.DataFrame(gs.cv_results_)
                append(TUNING, [{"k": k, "model": name, "best_params": json.dumps(best),
                                 "cv_auc_mean": gs.best_score_,
                                 "cv_auc_std": cvr.loc[gs.best_index_, "std_test_score"],
                                 "grid": json.dumps({str(p): round(s, 4) for p, s in
                                                     zip(cvr.params, cvr.mean_test_score)}),
                                 "seconds": round(time.time() - t, 1)}])
                print(f"k={k} {name:12s} CV AUC {gs.best_score_:.3f} best={best} ({time.time() - t:.0f}s)",
                      flush=True)

            # ---- final fits: one per seed (new balanced subsample + model seed), plus a shuffled-label control
            for kind, seed in [("model", s) for s in SEEDS] + [("shuffled", 0)]:
                if (str(k), name, kind, str(seed)) in runs_done:
                    continue
                t = time.time()
                idx = balanced_idx(y, N_TRAIN, seed=seed)
                ytr = y[idx].copy()
                if kind == "shuffled":
                    np.random.default_rng(123).shuffle(ytr)
                m = make_model(name, seed=seed).set_params(**best)
                m.fit(X[idx], ytr)
                row = {"k": k, "model": name, "kind": kind, "seed": seed, "params": json.dumps(best)}
                row.update(evaluate_all(lambda Z: (m.predict(Z), scores(m, Z)), M, test_bal))
                row["seconds"] = round(time.time() - t, 1)
                append(RUNS, [row])
                if name == "LogReg" and kind == "model" and seed == 0:
                    joblib.dump(m, MODELS_DIR / f"LogReg_k{k}_seed0.joblib")
                print(f"k={k} {name:12s} {kind:8s} seed={seed} human F1 {row['human_f1']:.3f} "
                      f"AUC {row['human_auc']:.3f} | mouse acc {row['mouse_acc']:.3f} AUC {row['mouse_auc']:.3f} "
                      f"({row['seconds']:.0f}s)", flush=True)
        del M, X
    permutation_null()
    print(f"\nALL DONE in {(time.time() - t_all) / 60:.1f} min", flush=True)


N_PERM = 10


def permutation_null():
    """A single shuffled-label fit can land a few points from 0.5 by chance; 10 permutations
    (LogReg, every k, tuned params) show the spread of the null distribution."""
    tune = pd.read_csv(TUNING)
    for k in K_VALUES:
        done = done_keys(RUNS, ["k", "model", "kind", "seed"])
        todo = [p for p in range(N_PERM) if (str(k), "LogReg", "perm_null", str(1000 + p)) not in done]
        if not todo:
            continue
        M = clean_matrices(k)
        X, y, _ = M["human_train"]
        yh = M["human_test"][1]
        test_bal = balanced_idx(yh, 2 * int(min(np.bincount(yh))), seed=0)
        best = json.loads(tune[(tune.k == k) & (tune.model == "LogReg")].iloc[0].best_params)
        idx = balanced_idx(y, N_TRAIN, seed=0)
        rows = []
        for p in todo:
            ytr = y[idx].copy()
            np.random.default_rng(1000 + p).shuffle(ytr)
            m = make_model("LogReg", seed=0).set_params(**best).fit(X[idx], ytr)
            row = {"k": k, "model": "LogReg", "kind": "perm_null", "seed": 1000 + p, "params": json.dumps(best)}
            row.update(evaluate_all(lambda Z: (m.predict(Z), scores(m, Z)), M, test_bal))
            row["seconds"] = 0.0
            rows.append(row)
        append(RUNS, rows)
        aucs = [r["human_auc"] for r in rows]
        print(f"k={k} LogReg permutation null: human AUC {np.mean(aucs):.3f} ± {np.std(aucs):.3f} "
              f"(range {min(aucs):.3f}-{max(aucs):.3f})", flush=True)


if __name__ == "__main__":
    main()
