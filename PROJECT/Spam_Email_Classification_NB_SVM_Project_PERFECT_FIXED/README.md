# Spam Email Classification Using Naive Bayes and SVM

A complete machine-learning web project for Spam/Ham classification using **Multinomial Naive Bayes** and **Support Vector Machine (SVM)**, with model comparison, SVM kernel analysis, hyperparameter tuning, and a responsive cream-and-blue web interface.

## Dataset variables

The project uses the two required dataset columns:

- **X = `v2`** — message text / input feature
- **Y = `v1`** — Spam/Ham target variable

The bundled dataset contains **5,572 raw messages**. After duplicate removal, it contains **5,169 messages: 4,516 Ham and 653 Spam**.

Two copies are included:

- `data/spam.csv` — working dataset
- `data/spam_canonical.csv` — clean local recovery copy

If `spam.csv` is missing, malformed, or from an older five-column version, the project automatically restores it from `spam_canonical.csv`. Normal use does not require downloading the dataset.

## Included features

- Live Spam/Ham classification for newly entered messages
- Naive Bayes and optimized SVM predictions side-by-side
- TF-IDF unigram + bigram feature extraction
- Accuracy, Precision, Recall, F1-Score, Specificity, ROC-AUC, FPR and FNR
- Naive Bayes and SVM confusion matrices
- SVM kernel comparison: Linear, RBF, Polynomial and Sigmoid
- Hyperparameter comparison for `C`, `gamma` and `degree`
- Stratified cross-validation ranked primarily by F1-score
- Automatic selection of the strongest tested SVM configuration
- SVM decision margin and decision-strength visualization
- Influential-word analysis and message profile
- Flask backend + HTML/CSS/JavaScript frontend

The SVM intentionally uses the stable public `predict()` and `decision_function()` APIs. It does **not** depend on SVC probability internals, which avoids cross-version failures such as `_effective_probability` errors.

## Windows quick start

1. Extract the ZIP into a **new folder**.
2. Close CMD windows running older copies of this project.
3. Double-click **`setup_and_run.bat`**.
4. The launcher creates an isolated `.venv` for this project and installs the required packages there.
5. A preflight check validates the dataset and ML artifacts. If the packaged artifacts do not match your installed Python/scikit-learn runtime, they are retrained automatically before the website starts.
6. The correct localhost page opens automatically.

The app prefers port `5000`. If that port is already being used, it selects the next free port and opens the correct address.

## Run after first setup

Double-click:

```text
run.bat
```

## Manual CMD setup

```cmd
cd path\to\project
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade -r requirements.txt
.venv\Scripts\python.exe preflight.py
.venv\Scripts\python.exe app.py
```

## Project structure

```text
project/
├── app.py
├── preflight.py
├── requirements.txt
├── setup_and_run.bat
├── run.bat
├── train_models.bat
├── data/
│   ├── spam.csv
│   ├── spam_canonical.csv
│   └── README.md
├── ml/
│   ├── artifacts.py
│   ├── data.py
│   ├── evaluation.py
│   ├── inference.py
│   ├── runtime.py
│   └── training.py
├── models/
├── results/
├── templates/
│   └── index.html
├── static/
│   ├── css/styles.css
│   └── js/app.js
└── tests/
```

## ML flow

```text
X = v2 message text                Y = v1 Spam/Ham label
        │                                  │
        └──────────────┬───────────────────┘
                       ↓
              Cleaning + duplicates
                       ↓
              80/20 stratified split
                       ↓
                    TF-IDF
                       ↓
          ┌────────────┴────────────┐
          ↓                         ↓
   Multinomial NB              Baseline SVM
          └────────────┬────────────┘
                       ↓
                Metric comparison
                       ↓
               SVM kernel study
        Linear / RBF / Poly / Sigmoid
                       ↓
             Hyperparameter tuning
                C / gamma / degree
                       ↓
                Optimized SVM
                       ↓
                Web classification
```

## Troubleshooting

**`python` is not recognized**  
Install Python 3.11 or newer and enable **Add Python to PATH**.

**The first setup takes some time**  
The preflight may retrain the ML experiment when your runtime differs from the packaged artifact runtime. On the reference environment the full experiment completed in about 24 seconds; laptop speed can vary.

**Port 5000 is already in use**  
Use the page opened automatically or the exact URL printed in CMD. The project searches ports 5000–5019.

**Dataset was accidentally edited**  
Run the project again. Invalid `data/spam.csv` is restored from the bundled `data/spam_canonical.csv`.

**A package was upgraded later**  
Run `setup_and_run.bat` again. Runtime metadata invalidates incompatible cached models and preflight retrains them locally.


## Technology stack

| Layer | Technologies |
| --- | --- |
| Backend and API | Python, Flask |
| Machine learning | scikit-learn: MultinomialNB, SVC, TfidfVectorizer |
| Data processing | pandas, NumPy |
| Model persistence | joblib |
| Frontend | HTML, CSS, JavaScript; cream-and-blue interface |
| Automated checks | pytest (install separately to run the tests) |

## Download and open in VS Code

Install Python 3.11 or newer with **Add Python to PATH** enabled. An internet connection is needed for the first package installation; the dataset is bundled locally.

Clone the repository:

```bash
git clone https://github.com/pranav269-commits/2520030477_ML.git
cd 2520030477_ML/PROJECT/Spam_Email_Classification_NB_SVM_Project_PERFECT_FIXED
```

Alternatively, use GitHub's **Code → Download ZIP**, extract it, and open the folder above in VS Code with **File → Open Folder**. This folder contains `app.py` and `requirements.txt`.

### Windows PowerShell / VS Code terminal

Run these commands from the project folder:

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe preflight.py
.\.venv\Scripts\python.exe app.py
```

Activation is optional because the commands invoke the virtual environment's Python directly. If `python` is unavailable but the Windows Python launcher is installed, use `py -3` for the first two commands.

Open the address printed in the terminal, normally **http://127.0.0.1:5000**. Keep the terminal open. Stop the server with **Ctrl+C**. On later runs, use `run.bat` or `.\.venv\Scripts\python.exe app.py`.

### Linux / macOS

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python preflight.py
.venv/bin/python app.py
```

### Use the website

1. Enter an email or message in the analysis box.
2. Submit it to see the Spam/Ham verdict and both model predictions.
3. Inspect the influential words, message profile, and SVM decision margin.
4. Explore the model metrics, confusion matrices, kernel comparisons, and hyperparameter results.

## How the implementation works

1. `ml/data.py` loads the bundled CSV, normalizes labels, removes duplicates, and recovers an invalid working dataset from its canonical copy.
2. `ml/training.py` performs an 80/20 stratified train/test split with `random_state=42`. TF-IDF learns its vocabulary on the training partition, with English stop-word removal, unigram/bigram features, sublinear term frequency, and up to 8,000 features in the full experiment.
3. Multinomial Naive Bayes uses `alpha=0.5`. The baseline SVM uses a linear kernel and `C=1.0`.
4. The kernel study compares linear, RBF, polynomial, and sigmoid SVMs. The full hyperparameter study evaluates ten configurations using two-fold stratified cross-validation on the training feature matrix and ranks them by F1-score, then recall, precision, and accuracy.
5. The selected SVM is fitted on the training partition and evaluated on the held-out test partition. Models are written to `models/` and experiment results to `results/summary.json`.
6. `ml/inference.py` transforms new text with the saved vectorizer and obtains predictions from NB and the selected SVM. If they disagree, the final verdict follows the selected SVM.
7. `app.py` serves the interface and JSON API. Preflight and startup recovery retrain missing or incompatible artifacts using the installed runtime.

## Bundled evaluation results

These values come from the uploaded `results/summary.json`, generated on September 11, 2026. They describe the bundled held-out test set of 1,034 messages; retraining may produce different values with a different runtime or dataset.

| Model | Accuracy | Precision | Recall | F1-score |
| --- | ---: | ---: | ---: | ---: |
| Multinomial Naive Bayes | 97.49% | 99.07% | 80.92% | 89.08% |
| Baseline linear SVM | 98.16% | 98.28% | 87.02% | 92.31% |
| Selected / optimized SVM | 98.07% | 97.44% | 87.02% | 91.94% |

The selected configuration is **RBF, C=10, gamma=0.1**, chosen by training cross-validation. The baseline linear SVM has a slightly higher F1-score on this particular held-out test set. “Optimized” refers to parameter selection and does not guarantee the highest test-set score.

## Backend endpoints

| Method | Route | Purpose |
| --- | --- | --- |
| GET | `/` | Web interface |
| GET | `/api/health` | Dataset/artifact readiness and runtime metadata |
| GET | `/api/summary` | Experiment results; initializes models if necessary |
| POST | `/api/analyze` | Analyze JSON such as `{"text": "Congratulations, claim your prize now!"}` |
| POST | `/api/retrain` | Retrain the complete experiment and return updated results |

Empty messages and messages longer than 15,000 characters are rejected.

## Retrain and run automated checks

From the project folder, run:

```powershell
.\.venv\Scripts\python.exe -m ml.training
```

Windows users can also double-click `train_models.bat`. To run the supplied tests:

```powershell
.\.venv\Scripts\python.exe -m pip install pytest
.\.venv\Scripts\python.exe -m pytest tests -q
```

On Linux/macOS, replace `.\.venv\Scripts\python.exe` with `.venv/bin/python`. Tests cover dataset handling, metrics, training, inference, packaged artifacts, and the frontend template contract.

## Interpretation and scope

- The dataset contains short SMS-style messages; results on long emails, other languages, or different sources may differ.
- SVM decision strength is a transformation of the decision margin, not a calibrated spam probability.
- Influential-word evidence comes from Naive Bayes token contributions. The message profile uses keyword heuristics.
- The application analyzes pasted text locally; it does not connect to an inbox or automatically filter incoming mail.
- The bundled server binds to localhost. Running this project does not publish a public website.
- The supplied `.gitignore` excludes virtual environments, Python caches, and future regenerated model/result files. The original packaged models and summary are included in this initial upload.
