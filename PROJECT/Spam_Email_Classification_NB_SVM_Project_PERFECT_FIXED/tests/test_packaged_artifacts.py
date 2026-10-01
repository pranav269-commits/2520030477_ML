from pathlib import Path
import json

from ml.inference import SpamClassifier

ROOT = Path(__file__).resolve().parents[1]


def test_packaged_summary_matches_final_dataset_and_full_svm_study():
    summary = json.loads((ROOT / "results" / "summary.json").read_text(encoding="utf-8"))
    dataset = summary["dataset"]
    assert dataset["raw_rows"] == 5572
    assert dataset["clean_rows"] == 5169
    assert dataset["ham"] == 4516
    assert dataset["spam"] == 653
    assert dataset["features"] == 8000
    assert dataset["variables"] == {
        "X": "v2 (message text)",
        "Y": "v1 (ham/spam target)",
    }
    assert len(summary["kernel_results"]) == 4
    assert len(summary["hyperparameter_results"]) >= 10


def test_packaged_models_load_and_classify_new_spam_and_ham_messages():
    classifier = SpamClassifier(ROOT / "models", ROOT / "results")
    spam = classifier.analyze(
        "URGENT congratulations you won a free cash prize click now to claim your reward"
    )
    ham = classifier.analyze(
        "Hi Pranav tomorrow machine learning class starts at 10 AM please bring the project report"
    )
    assert spam["naive_bayes"]["prediction"] == "spam"
    assert spam["svm"]["prediction"] == "spam"
    assert ham["naive_bayes"]["prediction"] == "ham"
    assert ham["svm"]["prediction"] == "ham"


def test_artifact_compatibility_rejects_wrong_runtime(tmp_path):
    from ml.artifacts import artifacts_ready
    from ml.runtime import ARTIFACT_SCHEMA

    models = tmp_path / "models"
    results = tmp_path / "results"
    models.mkdir()
    results.mkdir()
    for name in ["vectorizer.joblib", "naive_bayes.joblib", "svm_baseline.joblib", "svm_optimized.joblib"]:
        (models / name).write_bytes(b"placeholder")
    summary = {
        "artifact_schema": ARTIFACT_SCHEMA,
        "runtime": {"python": "0.0", "scikit_learn": "0.0"},
        "dataset": {
            "clean_rows": 5169, "ham": 4516, "spam": 653,
            "variables": {"X": "v2 (message text)", "Y": "v1 (ham/spam target)"},
        },
        "model_metrics": {"x": 1},
        "hyperparameter_results": [{"x": 1}],
        "best_params": {"kernel": "linear", "C": 1},
    }
    import json
    (results / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    assert artifacts_ready(models, results) is False
