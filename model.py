import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
data = pd.read_csv("sample_healthcare_data.csv")

# Rename columns to lowercase (if needed)
data.rename(columns={
    "Age": "age",
    "Blood_Level": "blood_level",
    "Pressure_Rate": "pressure_rate",
    "Sugar_Level": "sugar_level",
    "Glucose_Level": "glucose_level",
    "Risk": "risk"
}, inplace=True)

# Select features and target
X = data[["age", "blood_level", "pressure_rate", "sugar_level", "glucose_level"]]
y = data["risk"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save the trained model to a .pkl file
pickle.dump(model, open("risk_model.pkl", "wb"))

# Test the model (optional)
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Verify the saved model
loaded_model = pickle.load(open("risk_model.pkl", "rb"))
print(f"Loaded Model Test Accuracy: {loaded_model.score(X_test, y_test) * 100:.2f}%")
