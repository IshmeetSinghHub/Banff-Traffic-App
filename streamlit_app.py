import streamlit as st

# streamlit_app.py
# -----------------------------------------------------------
# Banff Traffic Management – XAI Demo App
#
# This app:
# - Shows a simple prediction section (placeholder)
# - Shows XAI visuals:
#     - Actual vs Predicted (demo)
#     - Feature Importance (demo)
#     - SHAP Explanation (text + placeholder)
#
# You can plug in your real model and real data later.
# -----------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# Optional: uncomment later when you use real SHAP
# import shap

# -----------------------------
# Streamlit page config
# -----------------------------
st.set_page_config(
    page_title="Banff Traffic XAI App",
    layout="wide"
)

# -----------------------------
# 1. Load model (optional)
# -----------------------------
@st.cache_resource
def load_model():
    """
    Try to load your trained model.
    EDIT the path below to match your repo.
    Example: models/final_model.joblib
    """
    model_path = "models/final_model.joblib"  # <<< CHANGE if needed

    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.warning(
            f"Could not load model from '{model_path}'. "
            f"This app will use demo data instead.\n\nError: {e}"
        )
        return None


# -----------------------------
# 2. Load data for XAI (optional)
# -----------------------------
@st.cache_resource
def load_data_for_xai():
    """
    Try to load some data for XAI plots.
    You can point this to your real test or validation data.
    For now, we just create demo data.
    """
    # TODO: replace this with something like:
    # df = pd.read_csv("data/banff_test_data.csv")
    # return df

    # --- Demo synthetic data (so the app always runs) ---
    np.random.seed(42)
    x = np.arange(0, 50)
    y_true = x + np.random.normal(0, 3, size=len(x))
    y_pred = x + np.random.normal(0, 3, size=len(x))

    df_demo = pd.DataFrame(
        {
            "y_true": y_true,
            "y_pred": y_pred,
            "day_of_week": np.random.choice(
                ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                size=len(x)
            ),
            "month": np.random.randint(1, 13, size=len(x)),
            "is_holiday": np.random.randint(0, 2, size=len(x)),
            "temp_c": np.random.normal(10, 8, size=len(x)),
            "parking_occupancy": np.random.randint(0, 100, size=len(x)),
            "lag_visitors": np.random.randint(0, 5000, size=len(x)),
        }
    )
    return df_demo


model = load_model()
df_xai = load_data_for_xai()

# -----------------------------
# 3. Page header
# -----------------------------
st.title("Banff Traffic Management – Explainable AI (XAI) Dashboard")

st.markdown(
    """
This app is built for the **Banff Traffic Management** project.

It shows:
- A simple **prediction section** (placeholder)
- **Explainable AI (XAI)** visuals:
  - Actual vs Predicted values
  - Feature importance
  - SHAP explanation (concept)

You can later connect this app directly to your real Banff model and data.
"""
)

st.divider()

# -----------------------------
# 4. Simple Prediction Section (placeholder)
# -----------------------------
st.header("🔢 Make a Prediction (Demo Inputs)")

st.write(
    "This section is a simple demo of how a user could input data "
    "to get a prediction from the model."
)

col1, col2 = st.columns(2)

with col1:
    day_of_week = st.selectbox("Day of Week", ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    month = st.selectbox("Month", list(range(1, 13)))
    is_holiday = st.selectbox("Is Holiday?", [0, 1])

with col2:
    temp_c = st.number_input("Temperature (°C)", -30.0, 40.0, 10.0)
    parking_occupancy = st.slider("Parking Occupancy (%)", 0, 100, 50)
    lag_visitors = st.number_input("Visitors Yesterday", 0, 50000, 1000)

input_df = pd.DataFrame(
    {
        "day_of_week": [day_of_week],
        "month": [month],
        "is_holiday": [is_holiday],
        "temp_c": [temp_c],
        "parking_occupancy": [parking_occupancy],
        "lag_visitors": [lag_visitors],
    }
)

st.write("**Input sent to the model (structure example):**")
st.dataframe(input_df)

if st.button("Predict (Demo)"):
    if model is None:
        st.info(
            "The real model is not loaded yet. "
            "Once you connect your trained pipeline, this button will show real predictions."
        )
    else:
        st.warning(
            "You need to encode these features the same way as in training "
            "(e.g., one-hot encoding for day_of_week) before prediction."
        )

st.divider()

# -----------------------------
# 5. XAI Section
# -----------------------------
st.header("🧠 Explainable AI (XAI)")

st.write(
    """
Here we explain how the model behaves using:
- **Actual vs Predicted** plot (model performance)
- **Feature Importance** (which features matter)
- **SHAP-style explanation** (why predictions move up or down)
"""
)

# Ensure df_xai has the columns we expect
if df_xai is None or "y_true" not in df_xai.columns or "y_pred" not in df_xai.columns:
    st.warning(
        "XAI demo data is not available or missing 'y_true'/'y_pred' columns. "
        "Please check the load_data_for_xai() function."
    )
else:
    # ---------------------------------------
    # 5.1 Actual vs Predicted
    # ---------------------------------------
    st.subheader("📈 Actual vs Predicted (Demo)")

    y_true = df_xai["y_true"].values
    y_pred = df_xai["y_pred"].values

    fig1, ax1 = plt.subplots()
    ax1.scatter(y_true, y_pred)
    line_min = min(y_true.min(), y_pred.min())
    line_max = max(y_true.max(), y_pred.max())
    ax1.plot([line_min, line_max], [line_min, line_max], "r--")
    ax1.set_xlabel("Actual")
    ax1.set_ylabel("Predicted")
    ax1.set_title("Actual vs Predicted (Demo Data)")
    st.pyplot(fig1)

    st.write(
        """
If the points are close to the red diagonal line,  
it means the model predictions are close to the real values.
"""
    )

    # ---------------------------------------
    # 5.2 Feature Importance (Demo)
    # ---------------------------------------
    st.subheader("🔎 Feature Importance (Demo)")

    # In a real tree-based model, you would use: model.feature_importances_
    # Here we just build a fake importance vector to show the idea.
    demo_features = [
        "day_of_week", "month", "is_holiday",
        "temp_c", "parking_occupancy", "lag_visitors"
    ]
    demo_importances = np.array([0.1, 0.05, 0.15, 0.3, 0.25, 0.15])

    sorted_idx = np.argsort(demo_importances)[::-1]
    sorted_features = np.array(demo_features)[sorted_idx]
    sorted_importances = demo_importances[sorted_idx]

    fig2, ax2 = plt.subplots()
    ax2.bar(sorted_features, sorted_importances)
    ax2.set_xticklabels(sorted_features, rotation=45, ha="right")
    ax2.set_ylabel("Importance")
    ax2.set_title("Feature Importance (Demo)")
    st.pyplot(fig2)

    st.write(
        """
In a real model:
- Features with **taller bars** have more impact on the prediction.
- You would replace this demo with `model.feature_importances_` and your actual feature names.
"""
    )

# ---------------------------------------
# 5.3 SHAP Explanation (Text + Placeholder)
# ---------------------------------------
st.subheader("🧩 SHAP Explanations (Concept)")

st.write(
    """
**SHAP (SHapley Additive exPlanations)** helps explain **why** a prediction is high or low.

For each prediction:
- Features with **positive SHAP values** push the prediction **higher**
- Features with **negative SHAP values** push the prediction **lower**

In your full version, you can:
1. Fit a SHAP explainer on your trained model  
2. Compute SHAP values for a sample of your Banff data  
3. Plot a SHAP summary plot inside Streamlit

For now, we only show the concept and structure here.
"""
)

st.info(
    "Once your real model and data are connected, you can replace the demo plots "
    "with true model-based XAI visuals."
)

# ---------------------------------------
# 6. Client-friendly Summary
# ---------------------------------------
st.divider()

st.header("📝 How This Helps the Client")

st.markdown(
    """
- **Actual vs Predicted plot** shows how accurate the model is.  
- **Feature Importance** shows which inputs (like day of week, temperature, or parking levels) matter most.  
- **SHAP-style explanations** (when added) show why a particular day’s prediction is high or low.

This makes the model more transparent and easier to trust for Banff traffic planning.
"""
)


