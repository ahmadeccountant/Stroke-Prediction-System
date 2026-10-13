from __future__ import annotations

import io
import zipfile
from typing import Any

import numpy as np
import pandas as pd
import requests
import streamlit as st
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATASET_PAGE_URL = (
    "https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset/data"
)
DATASET_DOWNLOAD_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "fedesoriano/stroke-prediction-dataset"
)

NUMERIC_FEATURES = [
    "age",
    "hypertension",
    "heart_disease",
    "avg_glucose_level",
    "bmi",
]
CATEGORICAL_FEATURES = [
    "gender",
    "ever_married",
    "work_type",
    "Residence_type",
    "smoking_status",
]


st.set_page_config(
    page_title="Stroke Risk Prediction System",
    page_icon=":heart:",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(ttl=3600, show_spinner=False)
def load_dataset() -> pd.DataFrame:
    """Download and parse the exact Kaggle dataset required by the assignment."""
    response = requests.get(DATASET_DOWNLOAD_URL, timeout=60)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        csv_name = next(
            (
                name
                for name in archive.namelist()
                if name.endswith("healthcare-dataset-stroke-data.csv")
            ),
            None,
        )
        if csv_name is None:
            raise RuntimeError(
                "The Kaggle archive did not contain the expected CSV file."
            )
        frame = pd.read_csv(archive.open(csv_name))

    required_columns = {
        "gender",
        "age",
        "hypertension",
        "heart_disease",
        "ever_married",
        "work_type",
        "Residence_type",
        "avg_glucose_level",
        "bmi",
        "smoking_status",
        "stroke",
    }
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        raise RuntimeError(f"Dataset is missing columns: {sorted(missing_columns)}")

    clean = frame.copy()
    clean["bmi"] = pd.to_numeric(clean["bmi"], errors="coerce")
    clean["stroke"] = clean["stroke"].astype(int)
    return clean


@st.cache_resource(show_spinner=False)
def train_model(frame: pd.DataFrame) -> dict[str, Any]:
    """Train the documented imputation, encoding, SMOTE, and Logistic Regression pipeline."""
    x = frame[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = frame["stroke"].copy()

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=619,
        stratify=y,
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )

    x_train_prepared = preprocessor.fit_transform(x_train)
    x_test_prepared = preprocessor.transform(x_test)

    smote = SMOTE(random_state=619)
    x_train_balanced, y_train_balanced = smote.fit_resample(
        x_train_prepared, y_train
    )

    classifier = LogisticRegression(max_iter=2000, random_state=619)
    classifier.fit(x_train_balanced, y_train_balanced)

    y_pred = classifier.predict(x_test_prepared)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    feature_names = preprocessor.get_feature_names_out()
    coefficient_values = classifier.coef_[0]
    impacts = pd.DataFrame(
        {
            "Feature": [
                name.replace("numeric__", "")
                .replace("categorical__", "")
                .replace("_", " ")
                .title()
                for name in feature_names
            ],
            "Coefficient": coefficient_values,
        }
    )
    impacts["Direction"] = np.where(
        impacts["Coefficient"] >= 0, "Higher stroke signal", "Lower stroke signal"
    )
    impacts["Magnitude"] = impacts["Coefficient"].abs()
    impacts = impacts.sort_values("Magnitude", ascending=False).head(10)

    return {
        "preprocessor": preprocessor,
        "classifier": classifier,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        "impacts": impacts,
        "train_rows_before_smote": len(x_train),
        "train_rows_after_smote": len(x_train_balanced),
        "holdout_rows": len(x_test),
    }


def risk_band(probability: float) -> tuple[str, str]:
    if probability >= 0.65:
        return (
            "High",
            "Discuss these results promptly with a qualified healthcare professional.",
        )
    if probability >= 0.35:
        return (
            "Moderate",
            "Consider a clinical review of modifiable risk factors with a healthcare professional.",
        )
    return (
        "Low",
        "The model estimates a lower risk; continue routine preventive care and professional guidance.",
    )


def render_patient_prediction(
    frame: pd.DataFrame, model: dict[str, Any]
) -> None:
    st.subheader("Patient risk prediction")
    st.caption(
        "Enter observed values only. This is an academic decision-support estimate, "
        "not a medical diagnosis."
    )

    with st.form("patient_prediction_form"):
        left, middle, right = st.columns(3)
        with left:
            gender = st.selectbox(
                "Gender", sorted(frame["gender"].dropna().unique().tolist())
            )
            age = st.number_input("Age", min_value=0.0, max_value=120.0, value=56.0)
            avg_glucose = st.number_input(
                "Average glucose level",
                min_value=0.0,
                value=108.0,
                step=0.1,
            )
            bmi_missing = st.checkbox("BMI unavailable", value=False)
            bmi = st.number_input(
                "BMI",
                min_value=5.0,
                max_value=100.0,
                value=27.4,
                step=0.1,
                disabled=bmi_missing,
            )
        with middle:
            hypertension = st.radio("Hypertension", [0, 1], format_func=lambda x: "Yes" if x else "No")
            heart_disease = st.radio("Heart disease", [0, 1], format_func=lambda x: "Yes" if x else "No")
            ever_married = st.selectbox(
                "Ever married", sorted(frame["ever_married"].dropna().unique().tolist())
            )
            residence_type = st.selectbox(
                "Residence type",
                sorted(frame["Residence_type"].dropna().unique().tolist()),
            )
        with right:
            work_type = st.selectbox(
                "Work type", sorted(frame["work_type"].dropna().unique().tolist())
            )
            smoking_status = st.selectbox(
                "Smoking status",
                sorted(frame["smoking_status"].dropna().unique().tolist()),
            )
            st.write("")
            st.write("")
            submitted = st.form_submit_button(
                "Estimate stroke risk", type="primary", use_container_width=True
            )

    if not submitted:
        st.info("Complete the profile and select Estimate stroke risk.")
        return

    patient = pd.DataFrame(
        [
            {
                "age": age,
                "hypertension": hypertension,
                "heart_disease": heart_disease,
                "avg_glucose_level": avg_glucose,
                "bmi": np.nan if bmi_missing else bmi,
                "gender": gender,
                "ever_married": ever_married,
                "work_type": work_type,
                "Residence_type": residence_type,
                "smoking_status": smoking_status,
            }
        ]
    )
    probability = float(
        model["classifier"].predict_proba(
            model["preprocessor"].transform(patient)
        )[0, 1]
    )
    band, recommendation = risk_band(probability)

    st.divider()
    result_left, result_right = st.columns([1, 2])
    with result_left:
        st.metric("Predicted stroke risk", f"{probability * 100:.1f}%")
        if band == "High":
            st.error(f"{band} risk")
        elif band == "Moderate":
            st.warning(f"{band} risk")
        else:
            st.success(f"{band} risk")
    with result_right:
        st.subheader("Model recommendation")
        st.write(recommendation)
        st.caption("Model: Logistic Regression trained with SMOTE-balanced Kaggle data.")


def main() -> None:
    st.title("Stroke Risk Prediction System")
    st.write(
        "An interpretable machine-learning workspace for exploring the Kaggle "
        "stroke dataset and estimating risk for an individual patient profile."
    )

    with st.sidebar:
        st.header("What this system does")
        st.write(
            "This system uses a patient's age, medical history, glucose level, BMI, "
            "and lifestyle details to estimate stroke risk as Low, Moderate, or High. "
            "Enter the patient's information in the prediction form to see the estimate. "
            "This result is not a medical diagnosis."
        )

    try:
        with st.spinner("Loading the exact Kaggle dataset and training the model..."):
            frame = load_dataset()
            model = train_model(frame)
    except Exception as error:
        st.error(f"Dataset or model loading failed: {error}")
        st.info("Check the internet connection and try again.")
        st.stop()

    positive_cases = int(frame["stroke"].sum())
    negative_cases = int(len(frame) - positive_cases)
    missing_bmi = int(frame["bmi"].isna().sum())

    st.subheader("Dataset summary")
    summary = st.columns(5)
    summary[0].metric("Rows", f"{len(frame):,}")
    summary[1].metric("Predictors", "10")
    summary[2].metric("Stroke-positive", f"{positive_cases:,}")
    summary[3].metric("Missing BMI", f"{missing_bmi:,}")
    summary[4].metric("Class ratio", f"{negative_cases / positive_cases:.1f}:1")

    st.subheader("Model evaluation")
    metrics = st.columns(3)
    metrics[0].metric("Accuracy", f"{model['accuracy'] * 100:.1f}%")
    metrics[1].metric("Precision", f"{model['precision'] * 100:.1f}%")
    metrics[2].metric("Recall", f"{model['recall'] * 100:.1f}%")

    evaluation_left, evaluation_right = st.columns(2)
    with evaluation_left:
        st.write("Confusion matrix")
        matrix = model["matrix"]
        st.dataframe(
            pd.DataFrame(
                [[matrix["TN"], matrix["FP"]], [matrix["FN"], matrix["TP"]]],
                index=["Actual: No stroke", "Actual: Stroke"],
                columns=["Predicted: No stroke", "Predicted: Stroke"],
            ),
            use_container_width=True,
        )
        st.caption(
            f"Training rows: {model['train_rows_before_smote']:,} before SMOTE, "
            f"{model['train_rows_after_smote']:,} after SMOTE. "
            f"Holdout rows: {model['holdout_rows']:,}."
        )
    with evaluation_right:
        st.write("Feature impacts")
        st.dataframe(
            model["impacts"][["Feature", "Coefficient", "Direction"]].assign(
                Coefficient=lambda values: values["Coefficient"].round(3)
            ),
            hide_index=True,
            use_container_width=True,
        )

    st.divider()
    render_patient_prediction(frame, model)

    st.divider()
    st.subheader("Dataset preview")
    st.dataframe(frame.head(10), use_container_width=True, hide_index=True)
    st.caption(
        "Source: exact Kaggle Healthcare Dataset Stroke Data. "
        "All calculations are performed using this automatically loaded dataset."
    )


if __name__ == "__main__":
    main()