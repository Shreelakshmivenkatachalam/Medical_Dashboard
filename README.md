**Healthcare Dashboard & Risk Prediction System<br>
Overview**<br>
This project provides an interactive Healthcare Dashboard and a Risk Prediction System built using Streamlit, Pandas, Plotly, Seaborn, Matplotlib, and Scikit-learn. <br> It allows users to explore healthcare data, visualize various features, and predict the healthcare risk of patients based on their data input.<br>
<br>
**Features:**<br>
Healthcare Dashboard:<br>

Visualize key statistics and data distributions.<br>
Filter the data by risk level.<br>
Display charts such as Risk Level Distribution, Age vs. Blood Level, and Blood Pressure Distribution.<br>
Show a Correlation Heatmap for numeric features in the dataset.<br>
Display Summary Statistics for selected data.<br>
<br>
**Risk Prediction System:**<br>

Input patient details such as age, blood level, pressure rate, sugar level, and glucose level.<br>
Predict the healthcare risk level (Low, Medium, High) based on input features using a pre-trained machine learning model (risk_model.pkl).<br>
Save the user input and predicted results into a JSON file for future reference.<br>
Installation<br>
<br>
![image](https://github.com/user-attachments/assets/0718e524-70b3-462e-b621-5b04468c4708)

**To run this project on your local machine, follow these steps:**<br>

1. Clone the Repository<br>

git clone https://github.com/your-username/healthcare-dashboard.git<br>
cd healthcare-dashboard<br>
<br>
2. Install Dependencies<br>
Create a virtual environment and install the required dependencies:<br>

pip install -r requirements.txt<br>
The requirements.txt should include the necessary libraries:<br>

streamlit<br>
pandas<br>
plotly<br>
seaborn<br>
matplotlib<br>
scikit-learn<br>
<br>
3. Prepare the Dataset<br>
Ensure you have the dataset file sample_healthcare_data.csv in the project directory.<br> Update the file_path variable in the code to point to the correct location of your dataset.<br>
<br>
4. Model File<br>
Make sure the trained model file (risk_model.pkl) is available in the project directory. This file is used to predict the healthcare risk of a patient based on input features.<br>
<br>
5. Run the Application<br>
After setting up the dataset and model, run the Streamlit app with the following command:<br>

streamlit run app.py<br>
<br>
**How It Works**<br>
Dashboard Page:<br>
Filters: You can filter the dataset by the risk level (Low, Medium, High) from the sidebar.<br>
Charts: Several charts are displayed to help visualize the dataset:<br>
Risk Level Distribution: A bar chart showing the count of each risk level.<br>
Age vs. Blood Level: A scatter plot displaying the relationship between age and blood level, colored by risk.<br>
Blood Pressure Distribution: A histogram for the blood pressure distribution.<br>
Correlation Heatmap: A heatmap that visualizes the correlation between numeric features.<br>
Summary Statistics: After the charts, the summary statistics of the filtered data are displayed.<br>
<br>
**Risk Prediction Page:**<br>
Inputs: The user can input various health data such as age, blood level, pressure rate, sugar level, and glucose level.<br>
Prediction: When the user clicks on the "Predict Risk" button, the app uses the trained model to predict the patient's risk level (Low, Medium, or High).<br>
Save Data: The user inputs and the predicted result are saved to a JSON file (patients.json) for future reference.<br>
<br>
**Files in the Project**<br>
app.py: The main application file containing the Streamlit dashboard and risk prediction system code.<br>
requirements.txt: List of dependencies required to run the project.<br>
sample_healthcare_data.csv: Example healthcare dataset (ensure this file is available).<br>
risk_model.pkl: A pre-trained machine learning model to predict healthcare risk (ensure this file is available).<br>
patients.json: A JSON file used to store the user's input and the predicted risk levels.<br>
<br>
**Future Improvements**<br>
More Prediction Models: Implement more prediction models and compare their performance.<br>
Data Source: Integrate with a real-time database or API to fetch patient data.<br>
User Authentication: Add user authentication for a more personalized experience.<br>
UI Enhancements: Improve UI/UX for better user engagement.<br>
