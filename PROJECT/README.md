# Spam Email Classification Using Naive Bayes and SVM

The complete runnable project is in [Spam_Email_Classification_NB_SVM_Project_PERFECT_FIXED](Spam_Email_Classification_NB_SVM_Project_PERFECT_FIXED/).

It includes a Flask web application, a cream-and-blue interface, bundled dataset, pretrained models, Naive Bayes/SVM comparisons, SVM kernel and hyperparameter experiments, evaluation results, and automated tests.

## Run on Windows

1. Install Python 3.11 or newer and enable **Add Python to PATH**.
2. Download this repository using **Code → Download ZIP** and extract it, or clone it with Git.
3. Open `PROJECT/Spam_Email_Classification_NB_SVM_Project_PERFECT_FIXED`.
4. Double-click **setup_and_run.bat** for first-time setup. It creates `.venv`, installs dependencies, checks the dataset/models, and starts the app.
5. Use the browser address printed in the terminal, normally **http://127.0.0.1:5000**. Keep the terminal open.
6. For subsequent runs, double-click **run.bat**. Stop with **Ctrl+C**.

## Run from VS Code / PowerShell

```powershell
git clone https://github.com/pranav269-commits/2520030477_ML.git
cd 2520030477_ML/PROJECT/Spam_Email_Classification_NB_SVM_Project_PERFECT_FIXED
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe preflight.py
.\.venv\Scripts\python.exe app.py
```

See the [complete project README](Spam_Email_Classification_NB_SVM_Project_PERFECT_FIXED/README.md) for the dataset, technology stack, ML implementation, metrics, API endpoints, Linux/macOS commands, retraining, tests, and troubleshooting.

The other files in this directory include the existing literature review, EDA notebook, abstract, presentation PDF, and original dataset.
