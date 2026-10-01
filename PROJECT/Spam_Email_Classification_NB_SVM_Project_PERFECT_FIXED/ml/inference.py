from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import numpy as np


class ArtifactCompatibilityError(RuntimeError):
    """Raised when cached ML artifacts cannot be safely used by this runtime."""


class SpamClassifier:
    def __init__(self, model_dir: Path | str, result_dir: Path | str):
        model_dir = Path(model_dir)
        result_dir = Path(result_dir)
        try:
            self.vectorizer = joblib.load(model_dir / "vectorizer.joblib")
            self.nb = joblib.load(model_dir / "naive_bayes.joblib")
            self.svm = joblib.load(model_dir / "svm_optimized.joblib")
            self.summary = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))
        except Exception as exc:  # joblib/sklearn errors vary by version
            raise ArtifactCompatibilityError(f"Cached ML artifacts could not be loaded: {exc}") from exc

        try:
            self.feature_names = np.asarray(self.vectorizer.get_feature_names_out())
            self.nb_log_odds = self.nb.feature_log_prob_[1] - self.nb.feature_log_prob_[0]
            self._validate_artifacts()
        except ArtifactCompatibilityError:
            raise
        except Exception as exc:
            raise ArtifactCompatibilityError(f"Cached ML artifacts are incompatible: {exc}") from exc

    def _validate_artifacts(self) -> None:
        """Exercise only stable public APIs so incompatibility is detected at load time."""
        try:
            sample = self.vectorizer.transform(["model compatibility check"])
            self.nb.predict(sample)
            self.nb.predict_proba(sample)
            self.svm.predict(sample)
            self.svm.decision_function(sample)
            if bool(self.svm.get_params().get("probability", False)):
                raise ArtifactCompatibilityError(
                    "Cached SVM uses legacy probability=True artifacts and must be retrained."
                )
        except ArtifactCompatibilityError:
            raise
        except Exception as exc:
            raise ArtifactCompatibilityError(f"Cached ML artifacts are incompatible: {exc}") from exc

    @staticmethod
    def _profile(text: str) -> dict:
        lower = text.lower()
        groups = {
            "urgency": ["urgent", "immediately", "now", "today", "limited", "final", "hurry"],
            "reward": ["winner", "won", "prize", "reward", "gift", "jackpot", "bonus"],
            "action": ["click", "call", "reply", "claim", "text", "visit", "verify"],
            "money": ["cash", "money", "£", "$", "₹", "pound", "dollar", "rupee"],
            "promotion": ["free", "offer", "exclusive", "discount", "deal", "entry", "voucher"],
        }
        scores = {}
        for name, terms in groups.items():
            hits = sum(lower.count(term) for term in terms)
            scores[name] = min(100, hits * 22)
        suspicious = sum(scores.values()) / max(len(scores), 1)
        scores["normality"] = max(0, min(100, round(100 - suspicious)))
        return {k: int(v) for k, v in scores.items()}

    @staticmethod
    def _decision_strength(margin: float) -> float:
        """Map absolute SVM margin to [0, 1] for visualization; this is not a probability."""
        absolute = abs(float(margin))
        return absolute / (1.0 + absolute)

    @staticmethod
    def _boundary_position(margin: float) -> float:
        """Map signed margin to [0, 1] purely as a visual position around the decision boundary."""
        strength = SpamClassifier._decision_strength(margin)
        return 0.5 + (0.5 * strength if margin >= 0 else -0.5 * strength)

    def _evidence(self, X, limit: int = 10) -> list[dict]:
        row = X.tocsr()[0]
        if row.nnz == 0:
            return []
        contributions = []
        for index, value in zip(row.indices, row.data):
            raw = float(value * self.nb_log_odds[index])
            contributions.append((index, raw))
        contributions.sort(key=lambda pair: abs(pair[1]), reverse=True)
        max_abs = max(abs(v) for _, v in contributions) or 1.0
        return [
            {
                "token": str(self.feature_names[index]),
                "direction": "spam" if value >= 0 else "ham",
                "strength": round(min(1.0, abs(value) / max_abs), 4),
                "contribution": round(value, 4),
            }
            for index, value in contributions[:limit]
        ]

    def analyze(self, text: str) -> dict:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Please enter an email or message before analysis.")
        if len(text) > 15_000:
            raise ValueError("Message is too long. Keep the analysis under 15,000 characters.")

        clean_text = re.sub(r"\s+", " ", text.strip())
        X = self.vectorizer.transform([clean_text])

        nb_prob = float(self.nb.predict_proba(X)[0, 1])
        nb_pred = "spam" if int(self.nb.predict(X)[0]) == 1 else "ham"

        # Intentionally avoid SVC.predict_proba(). Pickled probability-enabled SVC
        # objects are fragile across scikit-learn versions. The public, stable SVM
        # decision_function is sufficient for classification and decision strength.
        svm_pred = "spam" if int(self.svm.predict(X)[0]) == 1 else "ham"
        margin = float(np.ravel(self.svm.decision_function(X))[0])
        decision_strength = self._decision_strength(margin)
        boundary_position = self._boundary_position(margin)

        final = svm_pred if nb_pred != svm_pred else nb_pred
        return {
            "final_verdict": final,
            "agreement": nb_pred == svm_pred,
            "naive_bayes": {
                "prediction": nb_pred,
                "spam_probability": round(nb_prob, 6),
                "ham_probability": round(1 - nb_prob, 6),
                "confidence": round(max(nb_prob, 1 - nb_prob), 6),
            },
            "svm": {
                "prediction": svm_pred,
                "decision_strength": round(decision_strength, 6),
                "boundary_position": round(boundary_position, 6),
                "decision_margin": round(margin, 6),
                "kernel": self.summary["best_params"].get("kernel"),
                "C": self.summary["best_params"].get("C"),
                "gamma": self.summary["best_params"].get("gamma"),
                "degree": self.summary["best_params"].get("degree"),
            },
            "evidence_tokens": self._evidence(X),
            "message_profile": self._profile(clean_text),
            "message_stats": {
                "characters": len(clean_text),
                "words": len(clean_text.split()),
                "digits": sum(ch.isdigit() for ch in clean_text),
                "uppercase": sum(ch.isupper() for ch in clean_text),
                "punctuation": sum(ch in "!?$£₹%" for ch in clean_text),
            },
        }
