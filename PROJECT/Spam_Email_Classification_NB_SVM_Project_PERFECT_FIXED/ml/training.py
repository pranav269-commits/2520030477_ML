from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import ParameterGrid, StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

from .data import ensure_dataset, load_dataset
from .evaluation import classification_metrics
from .runtime import ARTIFACT_SCHEMA, runtime_signature

RANDOM_STATE = 42


def _scores_for(model, X_test, y_test) -> dict:
    pred = model.predict(X_test)
    # Prefer decision_function whenever it exists. This avoids touching the
    # version-sensitive SVC probability machinery while still providing a
    # valid ranking score for ROC-AUC. Naive Bayes falls back to predict_proba.
    decision = getattr(model, "decision_function", None)
    if callable(decision):
        score = decision(X_test)
    else:
        score = model.predict_proba(X_test)[:, 1]
    return classification_metrics(y_test, pred, score)


def _jsonable_params(params: dict) -> dict:
    return {k: (float(v) if isinstance(v, np.floating) else int(v) if isinstance(v, np.integer) else v) for k, v in params.items()}


def _kernel_model(kernel: str, C: float = 1.0, gamma="scale", degree: int = 3, probability: bool = False) -> SVC:
    return SVC(
        kernel=kernel,
        C=float(C),
        gamma=gamma,
        degree=int(degree),
        probability=probability,
        random_state=RANDOM_STATE,
    )


def run_kernel_study(X_train, y_train, X_test, y_test) -> list[dict]:
    results = []
    for kernel in ["linear", "rbf", "poly", "sigmoid"]:
        print(f"[ML] Kernel study: {kernel}", flush=True)
        model = _kernel_model(kernel)
        model.fit(X_train, y_train)
        metrics = _scores_for(model, X_test, y_test)
        results.append({"kernel": kernel, "C": 1.0, "gamma": "scale", "degree": 3 if kernel == "poly" else None, **metrics})
    return results


def _parameter_grid(quick: bool = False) -> list[dict]:
    if quick:
        spaces = [
            {"kernel": ["linear"], "C": [0.5, 1.0]},
            {"kernel": ["rbf"], "C": [1.0], "gamma": ["scale", 0.1]},
            {"kernel": ["poly"], "C": [1.0], "gamma": ["scale"], "degree": [2, 3]},
            {"kernel": ["sigmoid"], "C": [1.0], "gamma": ["scale"]},
        ]
    else:
        # Bounded but complete teaching grid: all four kernels are represented,
        # C is varied broadly, gamma is varied for nonlinear kernels, and
        # polynomial degree is explicitly compared. This keeps first-run
        # training practical while preserving the project's core experiment.
        spaces = [
            {"kernel": ["linear"], "C": [0.1, 1.0, 10.0]},
            {"kernel": ["rbf"], "C": [1.0, 10.0], "gamma": ["scale"]},
            {"kernel": ["rbf"], "C": [10.0], "gamma": [0.1]},
            {"kernel": ["poly"], "C": [1.0], "gamma": ["scale"], "degree": [2, 3]},
            {"kernel": ["sigmoid"], "C": [1.0, 10.0], "gamma": ["scale"]},
        ]
    return [params for space in spaces for params in ParameterGrid(space)]


def run_hyperparameter_study(X_train, y_train, quick: bool = False) -> list[dict]:
    """Cross-validate SVM parameter combinations and return a ranked table.

    The folds are evaluated manually so each fitted SVM computes predictions and
    decision scores only once per fold. This is substantially faster than the
    generic multi-scorer ``cross_validate`` path for nonlinear sparse-text SVMs.
    """
    cv_splits = 2
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    y_array = np.asarray(y_train, dtype=int)
    metric_names = ("accuracy", "precision", "recall", "f1", "specificity")

    rows = []
    grid = _parameter_grid(quick=quick)
    for index, params in enumerate(grid, start=1):
        print(f"[ML] Hyperparameter study {index}/{len(grid)}: {params}", flush=True)
        fold_metrics = []
        for train_index, valid_index in cv.split(X_train, y_array):
            model = _kernel_model(
                kernel=params["kernel"],
                C=params["C"],
                gamma=params.get("gamma", "scale"),
                degree=params.get("degree", 3),
            )
            model.fit(X_train[train_index], y_array[train_index])
            valid_X = X_train[valid_index]
            valid_y = y_array[valid_index]
            pred = model.predict(valid_X)
            fold_metrics.append(classification_metrics(valid_y, pred))

        row = {
            "kernel": params["kernel"],
            "C": float(params["C"]),
            "gamma": params.get("gamma"),
            "degree": int(params["degree"]) if "degree" in params else None,
        }
        for metric in metric_names:
            values = [fold[metric] for fold in fold_metrics if fold[metric] is not None]
            row[metric] = round(float(np.mean(values)), 6) if values else None
        rows.append(row)

    rows.sort(key=lambda r: (r["f1"], r["recall"], r["precision"], r["accuracy"]), reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def train_project(
    df: pd.DataFrame,
    model_dir: Path | str,
    result_dir: Path | str,
    quick: bool = False,
) -> dict:
    model_dir = Path(model_dir)
    result_dir = Path(result_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    raw_rows = int(df.attrs.get("raw_rows", len(df)))
    clean = df[["label", "message"]].copy()
    clean = clean[clean["label"].isin(["ham", "spam"])].reset_index(drop=True)
    if clean.empty or clean["label"].nunique() < 2:
        raise ValueError("Training data must contain non-empty ham and spam classes.")
    # Project notation: X is the input feature (v2/message) and Y is the target (v1/ham-spam).
    X = clean["message"].astype(str)
    Y = clean["label"].map({"ham": 0, "spam": 1}).astype(int)

    X_train_text, X_test_text, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=RANDOM_STATE, stratify=Y
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
        ngram_range=(1, 2),
        max_features=4000 if quick else 8000,
        sublinear_tf=True,
        min_df=1,
    )
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    nb = MultinomialNB(alpha=0.5)
    nb.fit(X_train, Y_train)
    nb_metrics = _scores_for(nb, X_test, Y_test)

    svm_baseline = _kernel_model("linear", C=1.0, probability=False)
    svm_baseline.fit(X_train, Y_train)
    svm_baseline_metrics = _scores_for(svm_baseline, X_test, Y_test)

    kernel_results = run_kernel_study(X_train, Y_train, X_test, Y_test)
    hyperparameter_results = run_hyperparameter_study(X_train, Y_train, quick=quick)
    best_params = {
        "kernel": hyperparameter_results[0]["kernel"],
        "C": hyperparameter_results[0]["C"],
    }
    if hyperparameter_results[0].get("gamma") is not None:
        best_params["gamma"] = hyperparameter_results[0]["gamma"]
    if hyperparameter_results[0].get("degree") is not None:
        best_params["degree"] = hyperparameter_results[0]["degree"]

    svm_optimized = _kernel_model(
        best_params["kernel"],
        best_params["C"],
        best_params.get("gamma", "scale"),
        best_params.get("degree", 3),
        probability=False,
    )
    svm_optimized.fit(X_train, Y_train)
    optimized_metrics = _scores_for(svm_optimized, X_test, Y_test)

    model_metrics = {
        "naive_bayes": nb_metrics,
        "svm_baseline": svm_baseline_metrics,
        "svm_optimized": optimized_metrics,
    }
    baseline_winner = max(
        [("Naive Bayes", nb_metrics["f1"]), ("SVM", svm_baseline_metrics["f1"])],
        key=lambda item: item[1],
    )[0]

    label_counts = clean["label"].value_counts().to_dict()
    summary = {
        "artifact_schema": ARTIFACT_SCHEMA,
        "runtime": runtime_signature(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "raw_rows": raw_rows,
            "clean_rows": int(len(clean)),
            "duplicates_removed": max(0, raw_rows - int(len(clean))),
            "ham": int(label_counts.get("ham", 0)),
            "spam": int(label_counts.get("spam", 0)),
            "train_rows": int(len(X_train_text)),
            "test_rows": int(len(X_test_text)),
            "features": int(len(vectorizer.get_feature_names_out())),
            "variables": {
                "X": "v2 (message text)",
                "Y": "v1 (ham/spam target)",
            },
        },
        "experiment": {
            "split": "80/20 stratified",
            "random_state": RANDOM_STATE,
            "feature_extraction": "TF-IDF (1-2 grams)",
            "selection_metric": "Cross-validated F1-score",
            "baseline_winner": baseline_winner,
        },
        "model_metrics": model_metrics,
        "kernel_results": kernel_results,
        "hyperparameter_results": hyperparameter_results,
        "best_params": _jsonable_params(best_params),
    }

    joblib.dump(vectorizer, model_dir / "vectorizer.joblib")
    joblib.dump(nb, model_dir / "naive_bayes.joblib")
    joblib.dump(svm_baseline, model_dir / "svm_baseline.joblib")
    joblib.dump(svm_optimized, model_dir / "svm_optimized.joblib")
    (result_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def train_from_disk(root: Path | str | None = None, quick: bool = False) -> dict:
    root = Path(root or Path(__file__).resolve().parents[1])
    dataset_path = ensure_dataset(root / "data" / "spam.csv")
    df = load_dataset(dataset_path)
    return train_project(df, root / "models", root / "results", quick=quick)


if __name__ == "__main__":
    result = train_from_disk()
    print("Training complete.")
    print("Best SVM parameters:", result["best_params"])
    print("Optimized SVM F1:", result["model_metrics"]["svm_optimized"]["f1"])
