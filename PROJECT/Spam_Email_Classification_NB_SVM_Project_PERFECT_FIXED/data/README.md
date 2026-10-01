# Dataset

This folder contains the project dataset in two columns:

- **X = `v2`** — message text / input feature
- **Y = `v1`** — Spam/Ham target variable

Files:

- `spam.csv` — working copy used by the ML pipeline
- `spam_canonical.csv` — clean bundled recovery copy

Both contain 5,572 raw messages. Duplicate removal during preprocessing produces 5,169 unique labelled messages: 4,516 Ham and 653 Spam.

If the working file becomes malformed, the loader restores it locally from `spam_canonical.csv` before training.
