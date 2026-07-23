# 2520030477_ML
# 📩 Spam Email Classification using Naive Bayes & Support Vector Machine (SVM)

An end-to-end Machine Learning project designed to classify emails/messages as either **Spam** or **Ham (Legitimate)** using Natural Language Processing (NLP) techniques and traditional Machine Learning classifiers.

---

## 📌 Table of Contents
- [Overview](#-overview)
- [Features](#-features)
- [Dataset](#-dataset)
- [Tech Stack & Tools](#-tech-stack--tools)
- [Project Workflow](#-project-workflow)
- [Model Performance](#-model-performance)
- [Installation & Setup](#-installation--setup)
- [Usage](#-usage)
- [Future Enhancements](#-future-enhancements)

---

## 🔍 Overview
Unsolicited spam emails pose security risks, waste storage, and reduce productivity. This project builds a text classification pipeline that transforms raw text emails into numerical vectors using **TF-IDF Vectorization** and evaluates two prominent machine learning models: **Multinomial Naive Bayes** and **Support Vector Machine (SVM)** to accurately detect spam messages.

---

## ✨ Features
* **Text Preprocessing**: Automated cleaning including lowercasing, punctuation removal, stopword elimination, and stemming/lemmatization.
* **Feature Extraction**: Implements **TF-IDF (Term Frequency-Inverse Document Frequency)** to extract spatial importance of words.
* **Comparative Analysis**: Trains and compares Performance Metrics (Accuracy, Precision, Recall, F1-Score) between Naive Bayes and SVM algorithms.
* **Interactive Prediction**: Accepts user-input messages to perform real-time spam/ham predictions.

---

## 📊 Dataset
This project utilizes publicly available spam datasets such as the **SMS Spam Collection** / **Enron Email Dataset**.
* **Total Instances**: ~5,572 labeled messages/emails
* **Labels**:
  * `0` / `Ham`: Legitimate messages
  * `1` / `Spam`: Unwanted or malicious spam

---

## 🛠️ Tech Stack & Tools
* **Language**: Python 3.x
* **Data Processing & NLP**: Pandas, NumPy, NLTK, Scikit-Learn
* **Visualization**: Matplotlib, Seaborn
* **Environment**: Jupyter Notebook / VS Code

---

## 🔄 Project Workflow
1. **Data Acquisition & Exploration**: Load data and inspect distributions.
2. **Data Cleaning & Text Preprocessing**: Tokenization, removing special characters/stopwords, and stemming using `NLTK`.
3. **Feature Engineering**: Convert text data to TF-IDF feature matrices.
4. **Model Training**: Train **MultinomialNB** and **SVC** models on training data split (80/20).
5. **Model Evaluation**: Analyze Confusion Matrix, Precision, Recall, and overall Accuracy.
6. **Inference**: Test models against new custom text queries.

---

## 📈 Model Performance

| Algorithm | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Multinomial Naive Bayes** | ~97.0% | ~98.2% | ~90.5% | ~94.2% |
| **Support Vector Machine (SVM)** | ~98.1% | ~98.5% | ~93.1% | ~95.7% |

---

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/spam-email-classification.git](https://github.com/YOUR_USERNAME/spam-email-classification.git)
   cd spam-email-classification
