# Resume Screening ML

Automated resume screening and candidate-job fit prediction using NLP and supervised learning (TF-IDF + XGBoost), deployed on Streamlit for real-time predictions.

## Overview

Recruiters sift through large volumes of CVs for graduate roles, making manual screening slow and inconsistent. This project classifies resumes as **Fit** or **Not Fit** for a given job role by comparing resume text against a job description using text-based features, producing a fit prediction and confidence score.

## Objective

To design and implement a machine learning system that automatically classifies whether a candidate's resume is a fit or not-fit for a given job role, based on textual similarity between the resume and job description.

## Goal

To reduce the manual effort recruiters spend screening large volumes of graduate resumes by building a model that ranks and filters candidates objectively using text-based features.

## Dataset

- Source: [med2425/resume-job-fit-merged-v1](https://huggingface.co/datasets/med2425/resume-job-fit-merged-v1) (Hugging Face)
- Contains paired resumes, job descriptions, and fit labels (`No Fit`, `Potential Fit`, `Good Fit`)
- Labels collapsed to binary: `No Fit` → 0, `Potential Fit`/`Good Fit` → 1

## Workflow

1. **Data Loading** — loaded from Hugging Face, saved locally for reproducibility
2. **Exploratory Data Analysis** — label distribution, resume/JD domain distribution, text length distribution
3. **Preprocessing** — lowercasing, punctuation/number removal, stopword removal, lemmatization
4. **Feature Engineering** — TF-IDF vectorization (5,000 features)
5. **Model Training** — compared Logistic Regression, Linear SVM, Random Forest, and XGBoost (default and class-weighted variants)
6. **Model Selection** — XGBoost (class-weighted) selected as best model based on F1 score
7. **Evaluation** — accuracy, precision, recall, F1, confusion matrix, ROC-AUC
8. **Deployment** — Streamlit app for real-time fit prediction

## Results

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.84 | 0.75 | 0.53 | 0.62 |
| Linear SVM (balanced) | 0.80 | 0.56 | 0.83 | 0.67 |
| Random Forest (balanced) | 0.89 | 0.90 | 0.63 | 0.74 |
| **XGBoost (weighted)** | **0.90** | **0.76** | **0.86** | **0.81** |

XGBoost was selected as the final model, achieving the best F1 score and an AUC of 0.957, indicating strong separation between fit and not-fit classes.

## Application

The deployed app allows a user to:
1. Select a role from a preset dropdown (or upload a custom job description)
2. Upload a resume (PDF or DOCX)
3. Receive an instant Fit / Not Fit prediction with a confidence score

**Live app:** [Resume Fit Checker](https://dammie-resume-fit-checker.streamlit.app/)
## Tech Stack

- Python, pandas, NumPy
- scikit-learn, XGBoost
- NLTK (text preprocessing)
- Streamlit (deployment)
- PyPDF2, python-docx (file parsing)

## Limitations

- Limited to text-based screening; does not parse resume layout, verify certifications, or assess soft skills
- Trained on a fixed set of job categories from the source dataset; may not generalize well to highly specialized or emerging roles
- Fit scoring is based on textual similarity rather than deep semantic understanding of experience quality — should be treated as a decision-support tool, not a replacement for human judgment

## Author

Damilola Odeshola
