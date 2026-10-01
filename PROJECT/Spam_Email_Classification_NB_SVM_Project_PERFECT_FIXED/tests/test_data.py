import pandas as pd
from ml.data import clean_dataset


def test_clean_dataset_keeps_v1_v2_and_removes_duplicates_and_invalid_rows():
    raw = pd.DataFrame({
        "v1": ["ham", "spam", "spam", "other", None],
        "v2": ["hello there", "free prize", "free prize", "ignore me", "missing"],
        "Unnamed: 2": [None] * 5,
    })
    clean = clean_dataset(raw)
    assert list(clean.columns) == ["label", "message"]
    assert clean.to_dict("records") == [
        {"label": "ham", "message": "hello there"},
        {"label": "spam", "message": "free prize"},
    ]


def test_ensure_dataset_replaces_malformed_existing_csv_with_uci_copy(tmp_path, monkeypatch):
    import io
    import zipfile
    import ml.data as data_module

    path = tmp_path / "spam.csv"
    # Reproduce the reported failure: 5-column header, 6-column data rows.
    path.write_bytes(b"v1,v2,,,\n" + (b'ham,"hello",,,,\n' * 9000))

    uci_text = "ham\tHello there\nspam\tWIN a free prize now\nham\tSee you tomorrow\nspam\tClaim reward urgently\n"
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("SMSSpamCollection", uci_text)

    monkeypatch.setattr(data_module, "_download", lambda url: payload.getvalue())

    ensured = data_module.ensure_dataset(path)
    clean = data_module.load_dataset(ensured)

    assert len(clean) == 4
    assert set(clean["label"]) == {"ham", "spam"}
    assert list(clean.columns) == ["label", "message"]


def test_bundled_dataset_matches_project_eda_counts():
    from pathlib import Path
    from ml.data import load_dataset

    project_root = Path(__file__).resolve().parents[1]
    clean = load_dataset(project_root / "data" / "spam.csv")

    assert clean.attrs["raw_rows"] == 5572
    assert len(clean) == 5169
    assert clean["label"].value_counts().to_dict() == {"ham": 4516, "spam": 653}


def test_ensure_dataset_restores_malformed_file_from_local_canonical_copy(tmp_path, monkeypatch):
    import ml.data as data_module

    working = tmp_path / "spam.csv"
    canonical = tmp_path / "spam_canonical.csv"
    working.write_text('v1,v2,,,\nham,"hello",,,,\n', encoding="utf-8")
    canonical.write_text(
        'v1,v2\n'
        'ham,"Hello there"\n'
        'spam,"WIN a free prize now"\n'
        'ham,"See you tomorrow"\n'
        'spam,"Claim reward urgently"\n',
        encoding="utf-8",
    )

    def no_network(*args, **kwargs):
        raise AssertionError("dataset recovery must not require network access")

    monkeypatch.setattr(data_module, "_download", no_network)
    ensured = data_module.ensure_dataset(working)
    clean = data_module.load_dataset(ensured)

    assert len(clean) == 4
    assert set(clean["label"]) == {"ham", "spam"}
    assert working.read_bytes() == canonical.read_bytes()


def test_bundled_source_csv_contains_only_v1_and_v2():
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[1]
    raw = pd.read_csv(project_root / "data" / "spam.csv", encoding="utf-8")
    assert list(raw.columns) == ["v1", "v2"]


def test_ensure_dataset_replaces_legacy_five_column_file_even_when_it_parses(tmp_path):
    import ml.data as data_module

    working = tmp_path / "spam.csv"
    canonical = tmp_path / "spam_canonical.csv"
    # Older copies of the Kaggle file have v1/v2 plus three unnamed columns.
    # The final project standard is deliberately a strict two-column source.
    working.write_text(
        'v1,v2,,,\n'
        'ham,"Hello there",,,\n'
        'spam,"WIN a prize",,,\n',
        encoding="utf-8",
    )
    canonical.write_text(
        'v1,v2\n'
        'ham,"Hello there"\n'
        'spam,"WIN a prize"\n',
        encoding="utf-8",
    )

    ensured = data_module.ensure_dataset(working)
    assert ensured.read_bytes() == canonical.read_bytes()
    assert list(pd.read_csv(ensured).columns) == ["v1", "v2"]


def test_ensure_dataset_recovers_from_two_column_parser_error_without_network(tmp_path, monkeypatch):
    import ml.data as data_module

    working = tmp_path / "spam.csv"
    canonical = tmp_path / "spam_canonical.csv"
    # Exact failure class: the header declares two fields but a later row has extras.
    working.write_text(
        'v1,v2\n'
        'ham,"Hello"\n'
        'spam,WIN,extra,fields\n',
        encoding="utf-8",
    )
    canonical.write_text(
        'v1,v2\n'
        'ham,"Hello"\n'
        'spam,"WIN a prize"\n',
        encoding="utf-8",
    )

    monkeypatch.setattr(
        data_module,
        "_download",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network must not be used")),
    )

    ensured = data_module.ensure_dataset(working)
    assert ensured.read_bytes() == canonical.read_bytes()
    assert list(pd.read_csv(ensured).columns) == ["v1", "v2"]
