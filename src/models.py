"""The 5 classical models and their small hyperparameter grids."""
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODEL_NAMES = ["LogReg", "KNN", "LinearSVM", "RandomForest", "HistGB"]

GRIDS = {
    "LogReg": {"logisticregression__C": [0.01, 0.1, 1.0]},
    "KNN": {"kneighborsclassifier__n_neighbors": [5, 15, 31]},
    "LinearSVM": {"sgdclassifier__alpha": [1e-3, 1e-2, 1e-1]},
    "RandomForest": {"min_samples_leaf": [1, 5], "max_features": ["sqrt", "log2"]},
    "HistGB": {"max_leaf_nodes": [15, 31]},
}

# parallel CV workers per model; RF and HistGB parallelize internally instead
CV_JOBS = {"LogReg": 4, "KNN": 4, "LinearSVM": 5, "RandomForest": 1, "HistGB": 1}


def make_model(name, seed=0, n_jobs=-1):
    # Frequencies are on very different scales across k-mers, so linear and
    # distance-based models get standardized inputs; trees don't need it.
    if name == "LogReg":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed))
    if name == "KNN":
        return make_pipeline(StandardScaler(), KNeighborsClassifier(n_jobs=n_jobs))
    if name == "LinearSVM":
        # Linear SVM (hinge loss) trained with SGD. LinearSVC took ~8 min per fit at
        # k=6 without converging, and an RBF kernel is infeasible at this data size.
        return make_pipeline(StandardScaler(), SGDClassifier(loss="hinge", max_iter=100, tol=1e-4,
                                                             random_state=seed))
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=200, n_jobs=n_jobs, random_state=seed)
    if name == "HistGB":
        return HistGradientBoostingClassifier(max_iter=100, early_stopping=False, random_state=seed)
    raise ValueError(name)


def scores(model, X):
    """Continuous score for ROC-AUC (probability if available, else decision function)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)
