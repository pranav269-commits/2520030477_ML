from pathlib import Path
import pandas as pd
from ml.training import train_project


def tiny_dataset():
    ham = [
        "meeting moved to room four", "please call me after class", "project report attached",
        "can we have lunch tomorrow", "your assignment has been received", "happy birthday friend",
        "lecture starts at nine", "send me the notes", "see you at the library", "thank you for helping me",
        "team meeting this evening", "please review the document",
    ]
    spam = [
        "win cash prize now", "claim your free reward today", "urgent winner click now",
        "free entry cash jackpot", "congratulations claim bonus", "exclusive prize call now",
        "you won money click link", "limited offer claim reward", "winner free gift urgent", "cash bonus text now",
        "claim your jackpot today", "free voucher winner",
    ]
    return pd.DataFrame({"label": ["ham"] * len(ham) + ["spam"] * len(spam), "message": ham + spam})


def test_train_project_writes_models_and_result_schema(tmp_path: Path):
    artifacts = train_project(
        tiny_dataset(),
        model_dir=tmp_path / "models",
        result_dir=tmp_path / "results",
        quick=True,
    )
    assert (tmp_path / "models" / "vectorizer.joblib").exists()
    assert (tmp_path / "models" / "naive_bayes.joblib").exists()
    assert (tmp_path / "models" / "svm_optimized.joblib").exists()
    assert (tmp_path / "results" / "summary.json").exists()
    assert set(artifacts["model_metrics"].keys()) == {"naive_bayes", "svm_baseline", "svm_optimized"}
    assert len(artifacts["kernel_results"]) == 4
    assert artifacts["hyperparameter_results"]
    assert "best_params" in artifacts


def test_train_project_reports_raw_and_clean_dataset_sizes(tmp_path: Path):
    data = tiny_dataset()
    data.attrs["raw_rows"] = len(data) + 3
    artifacts = train_project(
        data,
        model_dir=tmp_path / "models",
        result_dir=tmp_path / "results",
        quick=True,
    )
    assert artifacts["dataset"]["raw_rows"] == len(data) + 3
    assert artifacts["dataset"]["clean_rows"] == len(data)


def test_full_hyperparameter_grid_is_bounded_and_still_covers_kernel_controls():
    from ml.training import _parameter_grid

    rows = _parameter_grid(quick=False)
    assert len(rows) <= 17
    assert {row["kernel"] for row in rows} == {"linear", "rbf", "poly", "sigmoid"}
    assert {row["C"] for row in rows if row["kernel"] == "linear"} == {0.1, 1.0, 10.0}
    assert any(row.get("gamma") == 0.1 for row in rows if row["kernel"] in {"rbf", "sigmoid"})
    assert {row.get("degree") for row in rows if row["kernel"] == "poly"} == {2, 3}


def test_full_grid_avoids_pathological_high_c_cubic_polynomial_fit():
    from ml.training import _parameter_grid

    rows = _parameter_grid(quick=False)
    assert not any(
        row["kernel"] == "poly" and row["C"] == 10.0 and row.get("degree") == 3
        for row in rows
    )


def test_hyperparameter_study_does_not_use_slow_multimetric_cross_validate(monkeypatch):
    import ml.training as training
    from sklearn.feature_extraction.text import TfidfVectorizer

    data = tiny_dataset()
    y = data["label"].map({"ham": 0, "spam": 1}).astype(int).to_numpy()
    X = TfidfVectorizer().fit_transform(data["message"])

    def fail_cross_validate(*args, **kwargs):
        raise AssertionError("slow cross_validate path should not be used")

    monkeypatch.setattr(training, "cross_validate", fail_cross_validate)
    rows = training.run_hyperparameter_study(X, y, quick=True)
    assert rows
    assert all("f1" in row and "accuracy" in row for row in rows)


def test_training_summary_uses_X_for_v2_and_Y_for_v1(tmp_path: Path):
    artifacts = train_project(
        tiny_dataset(),
        model_dir=tmp_path / "models",
        result_dir=tmp_path / "results",
        quick=True,
    )
    assert artifacts["dataset"]["variables"] == {
        "X": "v2 (message text)",
        "Y": "v1 (ham/spam target)",
    }


def test_optimized_svm_is_trained_without_libsvm_probability_mode(tmp_path: Path):
    import joblib

    train_project(
        tiny_dataset(),
        model_dir=tmp_path / "models",
        result_dir=tmp_path / "results",
        quick=True,
    )
    model = joblib.load(tmp_path / "models" / "svm_optimized.joblib")
    assert model.get_params()["probability"] is False


def test_training_summary_records_runtime_compatibility_metadata(tmp_path: Path):
    import sys
    import sklearn

    artifacts = train_project(
        tiny_dataset(),
        model_dir=tmp_path / "models",
        result_dir=tmp_path / "results",
        quick=True,
    )
    assert artifacts["artifact_schema"] == 2
    assert artifacts["runtime"]["python"] == f"{sys.version_info.major}.{sys.version_info.minor}"
    assert artifacts["runtime"]["scikit_learn"] == sklearn.__version__
    assert {"numpy", "scipy", "joblib"}.issubset(artifacts["runtime"])
