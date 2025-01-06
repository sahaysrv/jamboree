import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Streamlit app setup
st.title("Admission Prediction App")
st.sidebar.header("Input Parameters")

# Function to get user input from the sidebar
def user_input_features():
    GRE_Score = st.sidebar.number_input("GRE Score", min_value=0, max_value=340, step=1, value=300)
    TOEFL_Score = st.sidebar.number_input("TOEFL Score", min_value=0, max_value=120, step=1, value=100)
    University_Rating = st.sidebar.slider("University Rating", min_value=1, max_value=5, step=1, value=3)
    SOP = st.sidebar.slider("SOP Strength", min_value=0.0, max_value=5.0, step=0.1, value=3.0)
    LOR = st.sidebar.slider("LOR Strength", min_value=0.0, max_value=5.0, step=0.1, value=3.0)
    CGPA = st.sidebar.number_input("CGPA", min_value=0.0, max_value=10.0, step=0.1, value=8.0)
    Research = st.sidebar.selectbox("Research Experience", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    # Return collected inputs as a dictionary
    return {
        "GRE_Score": GRE_Score,
        "TOEFL_Score": TOEFL_Score,
        "University_Rating": University_Rating,
        "SOP": SOP,
        "LOR": LOR,
        "CGPA": CGPA,
        "Research": Research,
    }

# Collect user inputs
input_data = user_input_features()

# Create a button to trigger predictions
if st.button("Predict"):
    # Convert user inputs to DataFrame
    data = CustomData(
        GRE_Score=int(input_data["GRE_Score"]),
        TOEFL_Score=float(input_data["TOEFL_Score"]),
        University_Rating=int(input_data["University_Rating"]),
        SOP=float(input_data["SOP"]),
        LOR=float(input_data["LOR"]),
        CGPA=float(input_data["CGPA"]),
        Research=int(input_data["Research"])
    )
    pred_df = data.get_data_as_data_frame()

    # Initialize prediction pipeline and make predictions
    predict_pipeline = PredictPipeline()
    results = predict_pipeline.predict(pred_df) * 100

    # Display results
    st.success(f"The predicted chance of admission is: {results[0]:.2f}%")

