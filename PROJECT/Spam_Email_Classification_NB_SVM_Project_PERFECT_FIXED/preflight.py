from __future__ import annotations

from pathlib import Path

from ml.artifacts import artifacts_ready
from ml.data import ensure_dataset, load_dataset
from ml.inference import ArtifactCompatibilityError, SpamClassifier
from ml.runtime import runtime_signature
from ml.training import train_from_disk

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "spam.csv"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"


def main() -> int:
    print("[CHECK] Runtime:", runtime_signature(), flush=True)
    dataset_path = ensure_dataset(DATA)
    clean = load_dataset(dataset_path)
    counts = clean["label"].value_counts().to_dict()
    if clean.attrs.get("raw_rows") != 5572 or len(clean) != 5169 or counts != {"ham": 4516, "spam": 653}:
        raise RuntimeError(f"Dataset validation failed: raw={clean.attrs.get('raw_rows')}, clean={len(clean)}, counts={counts}")
    print("[CHECK] Dataset: 5572 raw -> 5169 clean (4516 ham / 653 spam)", flush=True)

    needs_training = not artifacts_ready(MODELS, RESULTS)
    if not needs_training:
        try:
            SpamClassifier(MODELS, RESULTS)
        except Exception as exc:
            print(f"[CHECK] Cached model needs recovery: {exc}", flush=True)
            needs_training = True

    if needs_training:
        print("[CHECK] Training compatible local ML artifacts. This is done once for this environment...", flush=True)
        train_from_disk(ROOT, quick=False)

    classifier = SpamClassifier(MODELS, RESULTS)
    spam = classifier.analyze("URGENT winner claim your free cash prize now")
    ham = classifier.analyze("Hi, the machine learning project meeting is at 3 PM tomorrow")
    if spam["svm"]["prediction"] != "spam" or ham["svm"]["prediction"] != "ham":
        raise RuntimeError("Model smoke test failed to separate the bundled spam/ham examples.")
    print("[CHECK] Naive Bayes + SVM inference: OK", flush=True)
    print("[CHECK] Project preflight complete. Starting web server next.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
