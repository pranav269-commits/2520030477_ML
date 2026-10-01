from __future__ import annotations

from pathlib import Path
from urllib.request import Request, urlopen
from io import BytesIO, StringIO
import csv
import zipfile

import pandas as pd

DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/IntroToTextAnalyticsWithR/master/spam.csv"
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/228/sms%2Bspam%2Bcollection.zip"


def clean_dataset(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize supported spam datasets to label/message and remove duplicates."""
    candidates = [
        ("v1", "v2"),
        ("label", "message"),
        ("Category", "Message"),
        ("category", "message"),
    ]
    pair = next(((a, b) for a, b in candidates if a in raw.columns and b in raw.columns), None)
    if pair is None:
        raise ValueError("Dataset must contain v1/v2, label/message, or Category/Message columns.")

    label_col, message_col = pair
    raw_rows = int(len(raw))
    df = raw[[label_col, message_col]].copy()
    df.columns = ["label", "message"]
    df = df.dropna(subset=["label", "message"])

    # Match the project's EDA notebook exactly: duplicates are identified on
    # the original label/message values before whitespace normalization.
    # Stripping first silently merged 11 additional messages in spam.csv and
    # changed the documented 5,169-row / 653-spam distribution.
    df = df.drop_duplicates(subset=["label", "message"], keep="first")
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["message"] = df["message"].astype(str).str.strip()
    df = df[df["label"].isin(["ham", "spam"])]
    df = df[df["message"].str.len() > 0].reset_index(drop=True)
    df.attrs["raw_rows"] = raw_rows
    return df


def load_dataset(path: Path | str) -> pd.DataFrame:
    path = Path(path)
    last_error = None
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            raw = pd.read_csv(path, encoding=encoding)
            return clean_dataset(raw)
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Could not read dataset: {path}")


def _download(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0 ML-College-Project"})
    with urlopen(request, timeout=45) as response:
        return response.read()


def _uci_zip_to_csv(payload: bytes) -> bytes:
    with zipfile.ZipFile(BytesIO(payload)) as archive:
        member = next((name for name in archive.namelist() if name.endswith("SMSSpamCollection")), None)
        if member is None:
            raise RuntimeError("UCI archive does not contain SMSSpamCollection.")
        text = archive.read(member).decode("utf-8", errors="replace")
    output = StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(["v1", "v2"])
    for line in text.splitlines():
        if "\t" not in line:
            continue
        label, message = line.split("\t", 1)
        writer.writerow([label, message])
    return output.getvalue().encode("utf-8")


def _dataset_is_valid(path: Path) -> bool:
    """Return True only for the project's strict two-column v1/v2 CSV schema."""
    header_ok = False
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            header = pd.read_csv(path, encoding=encoding, nrows=0)
            header_ok = list(header.columns) == ["v1", "v2"]
            break
        except UnicodeDecodeError:
            continue
        except Exception:
            return False
    if not header_ok:
        return False

    try:
        clean = load_dataset(path)
    except Exception:
        return False
    return len(clean) >= 2 and {"ham", "spam"}.issubset(set(clean["label"]))


def _write_if_valid(path: Path, payload: bytes) -> bool:
    """Write a candidate CSV atomically enough for this local project, then validate it."""
    candidate = path.with_suffix(path.suffix + ".download")
    candidate.write_bytes(payload)
    if not _dataset_is_valid(candidate):
        candidate.unlink(missing_ok=True)
        return False
    candidate.replace(path)
    return True


def ensure_dataset(path: Path | str) -> Path:
    """Ensure the working dataset is valid, restoring it from the bundled local copy when needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and _dataset_is_valid(path):
        return path

    # A clean two-column copy ships with the project under a different filename.
    # Keeping it separate means an old/malformed spam.csv left from an earlier
    # extraction can always be replaced without any internet connection.
    canonical = path.with_name("spam_canonical.csv")
    if canonical.exists() and _dataset_is_valid(canonical):
        path.write_bytes(canonical.read_bytes())
        if _dataset_is_valid(path):
            return path

    errors = []

    # Emergency fallback only. Normal project use never needs the network because
    # spam_canonical.csv is bundled in the ZIP.
    try:
        payload = _uci_zip_to_csv(_download(UCI_ZIP_URL))
        if _write_if_valid(path, payload):
            return path
        errors.append("UCI archive converted to a CSV that did not validate")
    except Exception as exc:  # pragma: no cover - network varies by environment
        errors.append(f"UCI archive: {exc}")

    try:
        payload = _download(DATA_URL)
        if _write_if_valid(path, payload):
            return path
        errors.append("GitHub mirror CSV did not validate")
    except Exception as exc:  # pragma: no cover - network varies by environment
        errors.append(f"GitHub mirror: {exc}")

    raise RuntimeError(
        "The project dataset could not be loaded. Restore data/spam_canonical.csv from the project ZIP. "
        "Fallback attempts: " + " | ".join(errors)
    )

