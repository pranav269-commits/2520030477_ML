from pathlib import Path
import pandas as pd
import pytest
from ml.training import train_project
from ml.inference import SpamClassifier


def _dataset():
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


def test_analyze_returns_dual_model_schema(tmp_path: Path):
    train_project(_dataset(), tmp_path / "models", tmp_path / "results", quick=True)
    classifier = SpamClassifier(tmp_path / "models", tmp_path / "results")
    result = classifier.analyze("urgent winner claim your free cash prize now")
    assert result["final_verdict"] in {"spam", "ham"}
    assert result["naive_bayes"]["prediction"] in {"spam", "ham"}
    assert result["svm"]["prediction"] in {"spam", "ham"}
    assert 0 <= result["naive_bayes"]["confidence"] <= 1
    assert 0 <= result["svm"]["decision_strength"] <= 1
    assert isinstance(result["svm"]["decision_margin"], float)
    assert result["evidence_tokens"]
    assert set(result["message_profile"]) == {"urgency", "reward", "action", "money", "promotion", "normality"}


def test_analyze_rejects_empty_text(tmp_path: Path):
    train_project(_dataset(), tmp_path / "models", tmp_path / "results", quick=True)
    classifier = SpamClassifier(tmp_path / "models", tmp_path / "results")
    with pytest.raises(ValueError):
        classifier.analyze("   ")


def test_analyze_never_calls_svc_predict_proba(tmp_path: Path, monkeypatch):
    """SVM inference must remain compatible across scikit-learn versions."""
    train_project(_dataset(), tmp_path / "models", tmp_path / "results", quick=True)
    classifier = SpamClassifier(tmp_path / "models", tmp_path / "results")

    def forbidden(*args, **kwargs):
        raise AssertionError("SVC.predict_proba must not be used during inference")

    monkeypatch.setattr(type(classifier.svm), "predict_proba", forbidden, raising=True)
    result = classifier.analyze("urgent winner claim your free cash prize now")

    assert result["svm"]["prediction"] in {"spam", "ham"}
    assert 0 <= result["svm"]["decision_strength"] <= 1
    assert "spam_probability" not in result["svm"]
    assert "probability_alignment" not in result
