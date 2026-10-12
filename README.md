# Heart Disease Prediction System

A machine learning project that predicts the likelihood of heart disease using patient health attributes. It compares Logistic Regression and Random Forest models and provides predictions through an interactive Streamlit web application.

> **Disclaimer:** This project is for educational purposes only and is not intended for medical diagnosis or treatment.

## Features

* Data preprocessing and missing-value handling
* Numerical feature scaling and categorical feature encoding
* Logistic Regression model
* Random Forest model
* Interactive prediction interface using Streamlit
* Prediction probabilities, where supported by the model

## Tech Stack

* Python
* Pandas and NumPy
* Scikit-learn
* Joblib
* Streamlit
* Matplotlib and Seaborn (for exploratory data analysis)

## Project Structure

```text
heart-disease-project/
├── app.py
├── course.ipynb
├── models/
│   ├── logistic_regression.joblib
│   └── random_forest.joblib
├── requirements.txt
└── README.md
```

## Installation

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd heart-disease-project
```

### 2. Create and activate a virtual environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Run the Application

Ensure the trained model files are present in the `models/` directory, then run:

```bash
python -m streamlit run app.py
```

Streamlit will provide a local URL in the terminal. Open that URL in your browser.

## Models

### Logistic Regression

A baseline classification model used to estimate the likelihood of heart disease.

### Random Forest

An ensemble learning model that combines multiple decision trees for classification.

Model performance should be evaluated using precision, recall, F1-score, and the confusion matrix, in addition to accuracy. Recall for the heart-disease class is particularly important when assessing missed positive cases.

## Dataset

The project uses the Cleveland heart disease dataset with patient attributes and a binary target variable. Refer to the original dataset source for licensing, provenance, and intended use.

## Future Improvements

* Add XGBoost model comparison
* Improve evaluation and threshold selection
* Add visualizations to the dashboard
* Add model explainability

## Author

**Pranav Raj**
B.Tech — Mathematics and Computing
IIIT Bhagalpur
