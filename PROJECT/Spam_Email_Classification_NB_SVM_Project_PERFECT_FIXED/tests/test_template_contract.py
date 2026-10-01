from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_template_contains_all_major_experience_sections():
    html = (ROOT / "templates" / "index.html").read_text(encoding="utf-8")
    required_ids = [
        'id="analyzer"', 'id="results"', 'id="performance"', 'id="confusion"',
        'id="kernel-lab"', 'id="hyperparameter-lab"', 'id="methodology"'
    ]
    for marker in required_ids:
        assert marker in html
    assert "Spam Email Classification Using Naive Bayes and SVM" in html


def test_styles_use_cream_blue_palette_and_serif_display_font():
    css = (ROOT / "static" / "css" / "styles.css").read_text(encoding="utf-8")
    assert "--cream:" in css
    assert "--blue:" in css
    assert "Georgia" in css or "Times New Roman" in css
    assert "#ff0000" not in css.lower()
    assert "#00ff00" not in css.lower()


def test_javascript_has_required_api_hooks():
    js = (ROOT / "static" / "js" / "app.js").read_text(encoding="utf-8")
    assert "/api/summary" in js
    assert "/api/analyze" in js


def test_hyperparameter_table_uses_cv_metrics_that_do_not_require_expensive_probability_scoring():
    html = (ROOT / "templates" / "index.html").read_text(encoding="utf-8")
    header = html.split('id="hyperTable"', 1)[1].split("</thead>", 1)[0]
    assert "Specificity" in header


def test_interface_uses_X_Y_dataset_notation():
    html = (ROOT / "templates" / "index.html").read_text(encoding="utf-8")
    assert "X = v2" in html
    assert "Y = v1" in html


def test_interface_does_not_present_svm_probability_as_if_calibrated():
    html = (ROOT / "templates" / "index.html").read_text(encoding="utf-8").lower()
    js = (ROOT / "static" / "js" / "app.js").read_text(encoding="utf-8").lower()
    assert "probability alignment" not in html
    assert 'result.svm.spam_probability' not in js
    assert "r²" not in html
    assert "r-square" not in html
    assert "rsquare" not in html
