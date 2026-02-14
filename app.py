import numpy as np
import pandas as pd
import joblib
import streamlit as st


st.set_page_config(page_title="CreditWise Loan Prediction", page_icon=":money_with_wings:", layout="wide")

# Defaults for lower-impact fields not shown in the compact form.
FEATURE_DEFAULTS = {
    "Coapplicant_Income": 5205.5,
    "Age": 40.0,
    "Dependents": 1.0,
    "Existing_Loans": 2.0,
    "Savings": 9880.5,
    "Collateral_Value": 24321.0,
    "Marital_Status": "Married",
    "Employer_Category": "Private",
}


@st.cache_resource
def load_artifacts():
    model = joblib.load("Logistic_Reg.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_columns = joblib.load("columns.pkl")
    return model, scaler, feature_columns


def build_feature_row(inputs: dict, feature_columns: list[str]) -> pd.DataFrame:
    row = {col: 0.0 for col in feature_columns}

    row["Applicant_Income"] = float(inputs["Applicant_Income"])
    row["Coapplicant_Income"] = float(FEATURE_DEFAULTS["Coapplicant_Income"])
    row["Age"] = float(FEATURE_DEFAULTS["Age"])
    row["Dependents"] = float(FEATURE_DEFAULTS["Dependents"])
    row["Existing_Loans"] = float(FEATURE_DEFAULTS["Existing_Loans"])
    row["Savings"] = float(FEATURE_DEFAULTS["Savings"])
    row["Collateral_Value"] = float(FEATURE_DEFAULTS["Collateral_Value"])
    row["Loan_Amount"] = float(inputs["Loan_Amount"])
    row["Loan_Term"] = float(inputs["Loan_Term"])
    row["Education_Level"] = 1.0 if inputs["Education_Level"] == "Graduate" else 0.0

    ohe_candidates = [
        f"Employment_Status_{inputs['Employment_Status']}",
        f"Marital_Status_{FEATURE_DEFAULTS['Marital_Status']}",
        f"Loan_Purpose_{inputs['Loan_Purpose']}",
        f"Property_Area_{inputs['Property_Area']}",
        f"Gender_{inputs['Gender']}",
        f"Employer_Category_{FEATURE_DEFAULTS['Employer_Category']}",
    ]
    for col in ohe_candidates:
        if col in row:
            row[col] = 1.0

    dti_ratio = float(inputs["DTI_Ratio"])
    credit_score = float(inputs["Credit_Score"])
    applicant_income = float(inputs["Applicant_Income"])

    row["DTI_Ratio_sq"] = dti_ratio ** 2
    row["Credit_Score_sq"] = credit_score ** 2
    row["Applicant_Income_Log"] = float(np.log1p(applicant_income))

    return pd.DataFrame([row], columns=feature_columns)


def apply_custom_style() -> None:
    st.markdown(
        """
        <style>
            :root {
                color-scheme: light;
            }
            [data-testid="stAppViewContainer"] {
                background:
                    radial-gradient(circle at 15% 10%, #f7d7b5 0%, transparent 40%),
                    radial-gradient(circle at 90% 20%, #dceefb 0%, transparent 35%),
                    linear-gradient(160deg, #f8fbff 0%, #eef7f2 100%);
            }
            [data-testid="stAppViewContainer"],
            [data-testid="stAppViewContainer"] * {
                color: #1f2937;
            }
            .main .block-container {
                max-width: 980px;
                padding-top: 2rem;
                padding-bottom: 2rem;
            }
            .hero {
                background: rgba(255, 255, 255, 0.8);
                border: 1px solid #d8e2dc;
                border-radius: 16px;
                padding: 1rem 1.2rem;
                margin-bottom: 1rem;
            }
            .hero h1 {
                margin: 0;
                color: #1a365d;
                font-size: 2.1rem;
            }
            .hero p {
                margin: 0.35rem 0 0 0;
                color: #2d3748;
            }
            [data-testid="stWidgetLabel"] p,
            [data-testid="stMarkdownContainer"] p,
            label,
            .stMetricLabel div {
                color: #111827 !important;
                font-weight: 600;
            }
            .stNumberInput input,
            .stSelectbox div[data-baseweb="select"] > div,
            .stTextInput input {
                color: #111827 !important;
                background: rgba(255, 255, 255, 0.92) !important;
                border-color: #94a3b8 !important;
            }
            button[data-testid="stNumberInputStepUp"],
            button[data-testid="stNumberInputStepDown"] {
                background: #e2e8f0 !important;
                border: 1px solid #94a3b8 !important;
                color: #0f172a !important;
            }
            button[data-testid="stNumberInputStepUp"]:hover,
            button[data-testid="stNumberInputStepDown"]:hover {
                background: #cbd5e1 !important;
            }
            button[data-testid="stNumberInputStepUp"] svg,
            button[data-testid="stNumberInputStepDown"] svg {
                fill: #0f172a !important;
            }
            .stSlider [data-baseweb="slider"] * {
                color: #111827 !important;
            }
            [data-testid="stInfo"] {
                background: rgba(221, 239, 255, 0.82);
                border: 1px solid #93c5fd;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    apply_custom_style()

    st.markdown(
        """
        <div class="hero">
            <h1>CreditWise</h1>
            <p>Fast loan approval check using only high-impact applicant inputs.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        model, scaler, feature_columns = load_artifacts()
    except Exception as ex:
        st.error(f"Failed to load model files: {ex}")
        return

    left, right = st.columns([1.1, 1])

    with left:
        applicant_income = st.number_input("Applicant Income", min_value=0.0, value=10548.0, step=500.0)
        loan_amount = st.number_input("Loan Amount", min_value=1000.0, value=21210.5, step=500.0)
        loan_term = st.slider("Loan Term (months)", min_value=6, max_value=240, value=48, step=6)
        credit_score = st.slider("Credit Score", min_value=300, max_value=900, value=678, step=1)
        dti_ratio = st.slider("DTI Ratio", min_value=0.00, max_value=1.20, value=0.34, step=0.01)

    with right:
        education_level = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
        employment_status = st.selectbox("Employment Status", ["Salaried", "Self-employed", "Unemployed", "Contract"])
        loan_purpose = st.selectbox("Loan Purpose", ["Home", "Education", "Car", "Personal", "Business"])
        property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        st.info("Less influential model fields are auto-filled with stable defaults.")

    if st.button("Predict", type="primary", use_container_width=True):
        inputs = {
            "Applicant_Income": applicant_income,
            "Loan_Amount": loan_amount,
            "Loan_Term": float(loan_term),
            "Credit_Score": float(credit_score),
            "DTI_Ratio": float(dti_ratio),
            "Education_Level": education_level,
            "Employment_Status": employment_status,
            "Loan_Purpose": loan_purpose,
            "Property_Area": property_area,
            "Gender": gender,
        }

        try:
            feature_df = build_feature_row(inputs, feature_columns)
            scaled_features = scaler.transform(feature_df)
            pred = int(model.predict(scaled_features)[0])
            proba = float(model.predict_proba(scaled_features)[0][1])
        except Exception as ex:
            st.error(f"Prediction failed: {ex}")
            return

        verdict = "Likely Approved" if pred == 1 else "Likely Rejected"
        color = "#1f7a4d" if pred == 1 else "#9b2c2c"
        st.markdown(f"### Result: <span style='color:{color}'>{verdict}</span>", unsafe_allow_html=True)
        st.metric("Approval Probability", f"{proba * 100:.2f}%")
        st.progress(max(0.0, min(1.0, proba)))


if __name__ == "__main__":
    main()
