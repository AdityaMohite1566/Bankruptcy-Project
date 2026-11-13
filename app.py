
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# --------------------------------
# 1. Load trained model and features
# --------------------------------

BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(BASE_DIR / "logistic_regression_model.pkl")
feature_names = joblib.load(BASE_DIR / "feature_names.pkl")

# --------------------------------
# 2. Page Configuration
# --------------------------------

st.set_page_config(
    page_title="Bankruptcy Prediction System",
    page_icon="📊",
    layout="centered"
)

# --------------------------------
# 3. Application Title
# --------------------------------

st.title("🏢 Bankruptcy Prediction System")

st.write(
    "This application predicts whether a company "
    "is likely to face bankruptcy based on its "
    "financial and operational risk factors."
)

st.divider()

# --------------------------------
# 4. User Input Section
# --------------------------------

st.subheader("Enter Company Details")

input_values = {}

for feature in feature_names:

    input_values[feature] = st.select_slider(
        feature.replace("_", " ").title(),
        options=[0.0, 0.5, 1.0],
        value=0.5
    )

st.divider()

# --------------------------------
# 5. Prediction Section
# --------------------------------

if st.button("Predict Bankruptcy", type="primary"):

    # Convert inputs into DataFrame
    input_df = pd.DataFrame(
        [input_values],
        columns=feature_names
    )

    # Predict class
    prediction = model.predict(input_df)[0]

    # Predict probabilities
    probabilities = model.predict_proba(input_df)[0]

    # --------------------------------
    # 6. Convert Numerical Output
    #    into Class Names
    # --------------------------------

    class_names = {
        0: "Non-Bankruptcy",
        1: "Bankruptcy"
    }

    predicted_label = class_names[int(prediction)]

    # Match probabilities with their classes
    probability_by_class = {
        int(label): float(probability)
        for label, probability in zip(
            model.classes_,
            probabilities
        )
    }

    # --------------------------------
    # 7. Display Prediction Result
    # --------------------------------

    st.divider()

    st.subheader("Prediction Result")

    if int(prediction) == 1:

        st.error("⚠️ Bankruptcy Predicted")

    else:

        st.success("✅ Non-Bankruptcy Predicted")

    st.markdown(
        f"### Predicted Class: {predicted_label}"
    )

    # --------------------------------
    # 8. Display Prediction Probabilities
    # --------------------------------

    st.subheader("Prediction Probabilities")

    non_bankruptcy_probability = (
        probability_by_class[0] * 100
    )

    bankruptcy_probability = (
        probability_by_class[1] * 100
    )

    st.write(
        f"**Non-Bankruptcy:** "
        f"{non_bankruptcy_probability:.2f}%"
    )

    st.progress(
        probability_by_class[0]
    )

    st.write(
        f"**Bankruptcy:** "
        f"{bankruptcy_probability:.2f}%"
    )

    st.progress(
        probability_by_class[1]
    )

    # --------------------------------
    # 9. Disclaimer
    # --------------------------------

    st.caption(
        "Note: This prediction is generated using "
        "a trained Logistic Regression model. "
        "It is an estimated prediction and not "
        "a guarantee of the company's financial outcome."
    )

