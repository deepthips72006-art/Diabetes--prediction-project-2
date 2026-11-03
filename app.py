
import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("logistic-model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Diabetes Prediction App")

st.write("Enter patient details below")

# Input features
Pregnancies = st.number_input("Pregnancies", min_value=0)
Glucose = st.number_input("Glucose", min_value=0)
BloodPressure = st.number_input("BloodPressure", min_value=0)
SkinThickness = st.number_input("SkinThickness", min_value=0)
Insulin = st.number_input("Insulin", min_value=0)
BMI = st.number_input("BMI", min_value=0.0)
DiabetesPedigreeFunction = st.number_input(
    "DiabetesPedigreeFunction", min_value=0.0
)
Age = st.number_input("Age", min_value=1)

if st.button("Predict"):

    input_data = pd.DataFrame([[
        Pregnancies,
        Glucose,
        BloodPressure,
        SkinThickness,
        Insulin,
        BMI,
        DiabetesPedigreeFunction,
        Age
    ]], columns=[
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age"
    ])

    scaled_data = scaler.transform(input_data)

    prediction = model.predict(scaled_data)

    if prediction[0] == 1:
        st.warning("Model predicts a positive result.")
    else:
        st.success("Model predicts a negative result.")
