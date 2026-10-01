from __future__ import annotations

import json
import socket
import threading
import webbrowser
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from ml.artifacts import artifacts_ready as artifacts_ready_on_disk
from ml.data import ensure_dataset
from ml.inference import ArtifactCompatibilityError, SpamClassifier
from ml.runtime import runtime_signature
from ml.training import train_from_disk

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "results"
SUMMARY_PATH = RESULT_DIR / "summary.json"
DATASET_PATH = ROOT / "data" / "spam.csv"

app = Flask(__name__)
_train_lock = threading.Lock()
_classifier: SpamClassifier | None = None
_dataset_checked = False


def artifacts_ready() -> bool:
    return artifacts_ready_on_disk(MODEL_DIR, RESULT_DIR)


def _ensure_local_dataset() -> None:
    global _dataset_checked
    if not _dataset_checked:
        ensure_dataset(DATASET_PATH)
        _dataset_checked = True


def _load_classifier() -> SpamClassifier:
    return SpamClassifier(MODEL_DIR, RESULT_DIR)


def ensure_ready(force: bool = False) -> SpamClassifier:
    global _classifier
    _ensure_local_dataset()

    if _classifier is not None and artifacts_ready() and not force:
        return _classifier

    with _train_lock:
        if not force and artifacts_ready():
            try:
                _classifier = _load_classifier()
                return _classifier
            except Exception:
                _classifier = None

        # Train with the user's installed scikit-learn version. This is the
        # recovery path for missing, stale, or cross-version cached artifacts.
        train_from_disk(ROOT, quick=False)
        _classifier = _load_classifier()
        return _classifier


def choose_port(start: int = 5000, attempts: int = 20) -> int:
    """Return the first free localhost port, avoiding an older server on port 5000."""
    for port in range(start, start + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError("No free local port was found between 5000 and 5019.")


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    try:
        _ensure_local_dataset()
        dataset_ok = DATASET_PATH.exists()
    except Exception:
        dataset_ok = False
    return jsonify({
        "ready": artifacts_ready(),
        "dataset_present": dataset_ok,
        "project": "Spam Email Classification Using Naive Bayes and SVM",
        "variables": {"X": "v2", "Y": "v1"},
        "runtime": runtime_signature(),
    })


@app.get("/api/summary")
def summary():
    try:
        ensure_ready()
        return jsonify(json.loads(SUMMARY_PATH.read_text(encoding="utf-8")))
    except Exception as exc:
        return jsonify({"error": f"ML setup failed: {exc}"}), 503


@app.post("/api/analyze")
def analyze():
    payload = request.get_json(silent=True) or {}
    text = payload.get("text", "")
    try:
        classifier = ensure_ready()
        return jsonify(classifier.analyze(text))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except ArtifactCompatibilityError:
        # One controlled self-healing retry in case files changed after startup.
        try:
            classifier = ensure_ready(force=True)
            return jsonify(classifier.analyze(text))
        except Exception as retry_exc:
            return jsonify({"error": f"Model recovery failed: {retry_exc}"}), 503
    except Exception as exc:
        return jsonify({"error": f"Analysis failed: {exc}"}), 503


@app.post("/api/retrain")
def retrain():
    try:
        ensure_ready(force=True)
        return jsonify(json.loads(SUMMARY_PATH.read_text(encoding="utf-8")))
    except Exception as exc:
        return jsonify({"error": f"Retraining failed: {exc}"}), 503


if __name__ == "__main__":
    port = choose_port()
    url = f"http://127.0.0.1:{port}"
    print("\nSpam Email Classification Using Naive Bayes and SVM")
    print("X = v2 (message input) | Y = v1 (ham/spam target)")
    print(f"Open: {url}")
    print("Keep this CMD window open while using the website.\n")
    threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    app.run(host="127.0.0.1", port=port, debug=False)
