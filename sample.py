import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import numpy as np
import json
import os

# Function to load model and initialize data
def load_model_and_data():
    model = pickle.load(open("risk_model.pkl", "rb"))

    # JSON file to store user inputs
    data_file = "patients.json"

    # Initialize the JSON file if it doesn't exist
    if not os.path.exists(data_file):
        with open(data_file, "w") as file:
            json.dump([], file)

    return model, data_file


# Main Dashboard Function
def medical_dashboard():
    # Load dataset
    file_path = "sample_healthcare_data.csv"  # Update with the correct path
    df = pd.read_csv(file_path)

    # Streamlit UI
    st.title("Medical Dashboard")
    st.sidebar.header("Filters")

    # Filter by Risk Level
    risk_filter = st.sidebar.multiselect("Select Risk Level:", df["Risk"].unique(), default=df["Risk"].unique())
    df_filtered = df[df["Risk"].isin(risk_filter)]

    # Layout for arranging charts horizontally
    col1, col2 = st.columns(2)

    # Risk Level Distribution
    with col1:
        st.subheader("Risk Level Distribution")
        risk_count = df_filtered["Risk"].value_counts()
        st.bar_chart(risk_count)

    # Scatter Plot: Age vs. Blood Level
    with col2:
        st.subheader("Age vs. Blood Level")
        fig1 = px.scatter(df_filtered, x="Age", y="Blood_Level", color="Risk", title="Age vs. Blood Level", width=400, height=300)
        st.plotly_chart(fig1)

    # Another row for remaining charts
    col3, col4 = st.columns(2)

    # Histogram: Blood Pressure Distribution
    with col3:
        st.subheader("Blood Pressure Distribution")
        fig2 = px.histogram(df_filtered, x="Pressure_Rate", nbins=20, title="Blood Pressure Distribution", width=400, height=300)
        st.plotly_chart(fig2)

    # Correlation Heatmap (Exclude Categorical Columns)
    with col4:
        st.subheader("Correlation Heatmap")
        numeric_df = df_filtered.select_dtypes(include=["number"])  # Select only numeric columns
        fig, ax = plt.subplots(figsize=(4, 3))  # Set smaller figure size
        corr = numeric_df.corr()  # Compute correlation on numeric data
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    # Moved Summary Stats Section to the bottom
    st.subheader("Summary Statistics")
    st.write(df_filtered.describe())

    st.sidebar.text("Dashboard by Shree")


# Risk Prediction Function
def risk_analysis():
    # Load model and data
    model, data_file = load_model_and_data()

    # Streamlit UI for risk prediction
    st.title("Healthcare Risk Prediction")

    # Input fields for user data
    age = st.number_input("Age", min_value=1, max_value=120, step=1)
    blood_level = st.number_input("Blood Level", min_value=0.0, max_value=20.0, step=0.1)
    pressure_rate = st.number_input("Pressure Rate", min_value=0.0, max_value=200.0, step=0.1)
    sugar_level = st.number_input("Sugar Level", min_value=0.0, max_value=20.0, step=0.1)
    glucose_level = st.number_input("Glucose Level", min_value=0.0, max_value=200.0, step=0.1)

    # Predict button
    if st.button("Predict Risk"):
        try:
            # Prepare feature array for prediction
            features = np.array([[age, blood_level, pressure_rate, sugar_level, glucose_level]])

            # Make prediction
            prediction = model.predict(features)

            # Ensure prediction is handled properly
            if isinstance(prediction, np.ndarray):
                prediction = prediction[0]

            # Map prediction to risk level
            risk_labels = ["Low", "Medium", "High"]
            if isinstance(prediction, str):
                risk = prediction  # If prediction is already a label
            else:
                risk = risk_labels[int(prediction)] if 0 <= int(prediction) < len(risk_labels) else "Unknown"

            # Display result
            st.success(f"Predicted Risk: {risk}")

            # Save the data to the JSON file
            new_entry = {
                "age": age,
                "blood_level": blood_level,
                "pressure_rate": pressure_rate,
                "sugar_level": sugar_level,
                "glucose_level": glucose_level,
                "risk": risk
            }

            with open(data_file, "r+") as file:
                current_data = json.load(file)
                current_data.append(new_entry)
                file.seek(0)
                json.dump(current_data, file, indent=4)

        except Exception as e:
            st.error(f"Error: {str(e)}")


# Sidebar for Page Navigation
page = st.sidebar.radio("Select a Page", ["Dashboard", "Risk Analysis"])

if page == "Dashboard":
    medical_dashboard()
elif page == "Risk Analysis":
    risk_analysis()
