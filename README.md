**Healthcare Dashboard & Risk Prediction System
Overview**
This project provides an interactive Healthcare Dashboard and a Risk Prediction System built using Streamlit, Pandas, Plotly, Seaborn, Matplotlib, and Scikit-learn. <br> It allows users to explore healthcare data, visualize various features, and predict the healthcare risk of patients based on their data input.<br>

**Features:**
Healthcare Dashboard:

Visualize key statistics and data distributions.
Filter the data by risk level.
Display charts such as Risk Level Distribution, Age vs. Blood Level, and Blood Pressure Distribution.
Show a Correlation Heatmap for numeric features in the dataset.
Display Summary Statistics for selected data.

**Risk Prediction System:**

Input patient details such as age, blood level, pressure rate, sugar level, and glucose level.
Predict the healthcare risk level (Low, Medium, High) based on input features using a pre-trained machine learning model (risk_model.pkl).
Save the user input and predicted results into a JSON file for future reference.
Installation

**To run this project on your local machine, follow these steps:**

1. Clone the Repository
bash
Copy
Edit
git clone https://github.com/your-username/healthcare-dashboard.git
cd healthcare-dashboard
2. Install Dependencies
Create a virtual environment and install the required dependencies:

bash
Copy
Edit
pip install -r requirements.txt
The requirements.txt should include the necessary libraries:

txt
Copy
Edit
streamlit
pandas
plotly
seaborn
matplotlib
scikit-learn
3. Prepare the Dataset
Ensure you have the dataset file sample_healthcare_data.csv in the project directory. Update the file_path variable in the code to point to the correct location of your dataset.

4. Model File
Make sure the trained model file (risk_model.pkl) is available in the project directory. This file is used to predict the healthcare risk of a patient based on input features.

5. Run the Application
After setting up the dataset and model, run the Streamlit app with the following command:

bash
Copy
Edit
streamlit run app.py
**How It Works**
Dashboard Page:
Filters: You can filter the dataset by the risk level (Low, Medium, High) from the sidebar.
Charts: Several charts are displayed to help visualize the dataset:
Risk Level Distribution: A bar chart showing the count of each risk level.
Age vs. Blood Level: A scatter plot displaying the relationship between age and blood level, colored by risk.
Blood Pressure Distribution: A histogram for the blood pressure distribution.
Correlation Heatmap: A heatmap that visualizes the correlation between numeric features.
Summary Statistics: After the charts, the summary statistics of the filtered data are displayed.
**Risk Prediction Page:**
Inputs: The user can input various health data such as age, blood level, pressure rate, sugar level, and glucose level.
Prediction: When the user clicks on the "Predict Risk" button, the app uses the trained model to predict the patient's risk level (Low, Medium, or High).
Save Data: The user inputs and the predicted result are saved to a JSON file (patients.json) for future reference.
**Files in the Project**
app.py: The main application file containing the Streamlit dashboard and risk prediction system code.
requirements.txt: List of dependencies required to run the project.
sample_healthcare_data.csv: Example healthcare dataset (ensure this file is available).
risk_model.pkl: A pre-trained machine learning model to predict healthcare risk (ensure this file is available).
patients.json: A JSON file used to store the user's input and the predicted risk levels.
**Future Improvements**
More Prediction Models: Implement more prediction models and compare their performance.
Data Source: Integrate with a real-time database or API to fetch patient data.
User Authentication: Add user authentication for a more personalized experience.
UI Enhancements: Improve UI/UX for better user engagement.
