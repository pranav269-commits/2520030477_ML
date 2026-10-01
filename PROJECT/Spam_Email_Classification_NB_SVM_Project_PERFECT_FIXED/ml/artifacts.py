from __future__ import annotations

import json
from pathlib import Path

from .runtime import ARTIFACT_SCHEMA, runtime_signature


def summary_is_compatible(summary_path: Path | str) -> bool:
    summary_path = Path(summary_path)
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        dataset = summary["dataset"]
        variables = dataset.get("variables", {})
        return (
            int(summary.get("artifact_schema", -1)) == ARTIFACT_SCHEMA
            and summary.get("runtime") == runtime_signature()
            and variables.get("X") == "v2 (message text)"
            and variables.get("Y") == "v1 (ham/spam target)"
            and int(dataset.get("clean_rows", 0)) == 5169
            and int(dataset.get("ham", 0)) == 4516
            and int(dataset.get("spam", 0)) == 653
            and bool(summary.get("model_metrics"))
            and bool(summary.get("hyperparameter_results"))
            and bool(summary.get("best_params"))
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False


def artifacts_ready(model_dir: Path | str, result_dir: Path | str) -> bool:
    model_dir = Path(model_dir)
    result_dir = Path(result_dir)
    required = [
        model_dir / "vectorizer.joblib",
        model_dir / "naive_bayes.joblib",
        model_dir / "svm_baseline.joblib",
        model_dir / "svm_optimized.joblib",
        result_dir / "summary.json",
    ]
    return all(path.exists() and path.stat().st_size > 0 for path in required) and summary_is_compatible(
        result_dir / "summary.json"
    )
