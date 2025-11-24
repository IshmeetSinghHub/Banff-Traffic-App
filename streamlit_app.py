import streamlit as st

st.write("Hello world!")

# app.py - Banff Traffic XAI App (Template)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Optional: uncomment these if you will actually use SHAP
# import shap

st.set_page_config(page_title="Banff Traffic XAI App", layout="wide")

# -----------------------------
# 1. Load Model (EDIT THIS)
# -----------------------------
@st.cache_resource
def load_model():
    # TODO: change this path to your real model file
    # Example: "models/random_forest_banff.joblib"
    model_path = "models/final_model.joblib"
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Could not load model from {model_path}. Error: {e}")
        return None

model = load_model()

# -----------------------------
# 2. App Header
# -----------------------------
st.title("Banff Traffic Management – Explainable AI (XAI) Dashboard")

st.markdown(
    """
This app helps explain how our machine learning model makes predictions
for Banff traffic / visitors.

We show:
- **Predictions** based on input features  
- **Actual vs Predicted** plot  
- **Feature Importance**  
- (Optional) **SHAP explanations** for why predictions change  
"""
)

# -----------------------------
# 3. Example Input Section
# -----------------------------
st.header("🔢 Make a Prediction")

st.write("Enter input values to get a prediction from the model.")

col1, col2 = st.columns(2)

with col1:
    day_of_week = st.selectbox("Day of Week", ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    month = st.selectbox("Month", list(range(1, 13)))
    is_holiday = st.selectbox("Is Holiday?", [0, 1])

with col2:
    temp_c = st.number_input("Temperature (°C)", -30.0, 40.0, 10.0)
    parking_occupancy = st.slider("Parking Occupancy (%)", 0, 100, 50)
    lag_visitors = st.number_input("Visitors Yesterday", 0, 50000, 1000)

# Create a single-row DataFrame for prediction
input_data = pd.DataFrame(
    {
        "day_of_week": [day_of_week],
        "month": [month],
        "is_holiday": [is_holiday],
        "temp_c": [temp_c],
        "parking_occupancy": [parking_occupancy],
        "lag_visitors": [lag_visitors],
    }
)

st.write("**Model input preview:**")
st.dataframe(input_data)

if model is not None:
    if st.button("Predict Visitors / Traffic"):
        try:
            # ⚠️ You may need to encode day_of_week or scale features same as training
            # For now we just show a placeholder
            st.warning("You still need to match this input format to your trained model pipeline.")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
else:
    st.error("Model not loaded. Please check file path in load_model().")

# -----------------------------
# 4. XAI Section
# -----------------------------
st.header("🧠 Explainable AI (XAI)")

st.markdown(
    """
Below are example explainability tools that can be connected to your model:

1. **Actual vs Predicted Plot** – shows how close the model predictions are to real values  
2. **Feature Importance** – shows which features the model uses the most  
3. **(Optional) SHAP Plots** – explain how each feature pushes the prediction up or down  
"""
)

# -----------------------------
# 4.1 Actual vs Predicted (Dummy Example)
# -----------------------------
st.subheader("📈 Actual vs Predicted (Example)")

st.write(
    "This plot is a placeholder using fake data. You will replace it with real y_true and y_pred from your model."
)

# Fake data for demo
x = np.arange(0, 50)
y_true = x + np.random.normal(0, 3, size=len(x))
y_pred = x + np.random.normal(0, 3, size=len(x))

fig1, ax1 = plt.subplots()
ax1.scatter(y_true, y_pred)
ax1.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--")
ax1.set_xlabel("Actual")
ax1.set_ylabel("Predicted")
ax1.set_title("Actual vs Predicted (Demo)")
st.pyplot(fig1)

# -----------------------------
# 4.2 Feature Importance (Dummy Example)
# -----------------------------
st.subheader("🔎 Feature Importance (Example)")

st.write(
    "If your model is a tree-based model (e.g., RandomForest), you can use `model.feature_importances_`."
)

demo_features = ["day_of_week", "month", "is_holiday", "temp_c", "parking_occupancy", "lag_visitors"]
demo_importances = np.array([0.1, 0.05, 0.15, 0.3, 0.25, 0.15])

fig2, ax2 = plt.subplots()
ax2.bar(demo_features, demo_importances)
ax2.set_xticklabels(demo_features, rotation=45, ha="right")
ax2.set_ylabel("Importance")
ax2.set_title("Feature Importance (Demo)")
st.pyplot(fig2)

st.info(
    "Later, replace the demo arrays with your real feature names and `model.feature_importances_` values."
)

# -----------------------------
# 4.3 SHAP Placeholder Text
# -----------------------------
st.subheader("🧩 SHAP Explanations (Placeholder)")

st.write(
    """
SHAP (SHapley Additive exPlanations) can be used with tree-based models to explain individual predictions.

Steps you can do later:
1. Fit `shap.TreeExplainer(model)`  
2. Compute SHAP values for a sample of your data  
3. Show SHAP summary plot in Streamlit  

For now, we only show this description as a placeholder.
"""
)

st.success("XAI section structure is ready. You can now plug in your real model and data.")
