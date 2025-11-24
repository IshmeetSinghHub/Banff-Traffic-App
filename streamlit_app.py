import streamlit as st

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

st.header("🧠 Explainable AI (XAI)")

st.write(
    """
To help the client understand the model, we show:
- How close predictions are to real values
- Which features are most important
- (Optional) Why individual predictions increase or decrease
"""
)

