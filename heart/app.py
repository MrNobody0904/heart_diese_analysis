
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ---------- Configuration ----------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"


@st.cache_resource
def load_model(model_name):
    model_path = MODEL_DIR / model_name
    return joblib.load(model_path)


# ---------- Header ----------
st.title("❤️ Heart Disease Prediction")
st.write(
    "An educational machine-learning project using "
    "the Cleveland heart disease dataset."
)

st.warning(
    "This app is for educational demonstration only. "
    "It is not a medical diagnostic tool and must not "
    "be used to make healthcare decisions."
)

# ---------- Model Selection ----------
model_options = {
    "Logistic Regression": "logistic_regression.joblib",
    "Random Forest": "random_forest.joblib"
}

selected_model = st.selectbox(
    "Choose a trained model",
    list(model_options.keys())
)

try:
    model = load_model(model_options[selected_model])
except Exception as error:
    st.error(
        f"Could not load the model: {error}. "
        "Check that the model file exists in the models folder "
        "and was saved using a compatible scikit-learn version."
    )
    st.stop()


# ---------- Input Form ----------
st.subheader("Enter feature values")

with st.form("patient_form"):
    age = st.number_input("Age (years)", 1, 120, 50)

    sex = st.selectbox(
        "Sex (dataset code)",
        [0, 1]
    )

    cp = st.selectbox(
        "Chest pain type (dataset code)",
        [0, 1, 2, 3]
    )

    trestbps = st.number_input(
        "Resting blood pressure (mm Hg)",
        50, 250, 120
    )

    chol = st.number_input(
        "Serum cholesterol (mg/dl)",
        50, 700, 200
    )

    fbs = st.selectbox(
        "Fasting blood sugar > 120 mg/dl (code)",
        [0, 1]
    )

    restecg = st.selectbox(
        "Resting ECG (dataset code)",
        [0, 1, 2]
    )

    thalach = st.number_input(
        "Maximum heart rate achieved",
        50, 250, 150
    )

    exang = st.selectbox(
        "Exercise-induced angina (code)",
        [0, 1]
    )

    oldpeak = st.number_input(
        "Oldpeak",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.1
    )

    slope = st.selectbox(
        "Slope (dataset code)",
        [0, 1, 2]
    )

    ca = st.selectbox(
        "Major vessels (ca)",
        [0, 1, 2, 3, 4]
    )

    thal = st.selectbox(
        "Thal (dataset code)",
        [0, 1, 2, 3]
    )

    submitted = st.form_submit_button("Predict")


# ---------- Prediction ----------
if submitted:
    input_data = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }])

    try:
        prediction = int(model.predict(input_data)[0])
        probability = float(
            model.predict_proba(input_data)[0][1]
        )

        st.subheader("Prediction result")

        if prediction == 1:
            st.error("Model prediction: Class 1")
        else:
            st.success("Model prediction: Class 0")

        st.metric(
            "Estimated probability of class 1",
            f"{probability:.1%}"
        )

        st.caption(
            "Class 0 and class 1 follow your training dataset's "
            "target encoding. The probability is a model estimate, "
            "not a clinical risk estimate."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")