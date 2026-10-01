from __future__ import annotations

import sys

import joblib
import numpy as np
import scipy
import sklearn

ARTIFACT_SCHEMA = 2


def runtime_signature() -> dict[str, str]:
    """Return dependency versions that can affect serialized sklearn artifacts."""
    return {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "scikit_learn": sklearn.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "joblib": joblib.__version__,
    }
