"""Metrics helpers."""
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score

from .models import scores


def evaluate(model, X, y):
    pred = model.predict(X)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y, pred),
        "f1": f1_score(y, pred),
        "roc_auc": roc_auc_score(y, scores(model, X)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def majority_baseline(y_train, y_eval):
    maj = int(np.bincount(y_train).argmax())
    pred = np.full(len(y_eval), maj)
    return {
        "accuracy": accuracy_score(y_eval, pred),
        "f1": f1_score(y_eval, pred, zero_division=0),
        "roc_auc": 0.5,
    }
