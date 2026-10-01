import math
from ml.evaluation import classification_metrics


def test_classification_metrics_perfect_case():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 1, 1]
    y_score = [0.02, 0.08, 0.91, 0.97]
    result = classification_metrics(y_true, y_pred, y_score)
    assert result["accuracy"] == 1.0
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1"] == 1.0
    assert result["specificity"] == 1.0
    assert result["fpr"] == 0.0
    assert result["fnr"] == 0.0
    assert result["confusion_matrix"] == [[2, 0], [0, 2]]
    assert math.isclose(result["roc_auc"], 1.0)


def test_classification_metrics_imperfect_case():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 0, 1]
    y_score = [0.2, 0.7, 0.4, 0.8]
    result = classification_metrics(y_true, y_pred, y_score)
    assert result["accuracy"] == 0.5
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5
    assert result["specificity"] == 0.5
    assert result["fpr"] == 0.5
    assert result["fnr"] == 0.5
    assert result["confusion_matrix"] == [[1, 1], [1, 1]]
