import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load data
data = pd.read_csv("data/student_data.csv")

# Remove rows with invalid or missing values
data = data.dropna()  # Drop rows with missing values
data = data[data["attendance"].apply(lambda x: str(x).isdigit())]  # Keep only numeric rows

# Convert columns to numeric (if necessary)
data["attendance"] = pd.to_numeric(data["attendance"])
data["homework_score"] = pd.to_numeric(data["homework_score"])
data["midterm_score"] = pd.to_numeric(data["midterm_score"])
data["final_score"] = pd.to_numeric(data["final_score"])

# Define the label_risk function
def label_risk(final_score):
    if (final_score < 50):
        return "High"
    elif (50 <= final_score <= 69):
        return "Medium"
    else:
        return "Low"

# Apply the function to create a risk level column
data["risk_level"] = data["final_score"].apply(label_risk)

# Define features and target
X = data[["attendance", "homework_score", "midterm_score", "final_score"]]
y = data["risk_level"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save model
with open("data/model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model training complete. Saved as 'data/model.pkl'.")

# Sample data
sample_data = pd.DataFrame({
    "attendance": [78, 45, 88],
    "homework_score": [65, 40, 90],
    "midterm_score": [70, 50, 85],
    "final_score": [72, 55, 87]
})

print(sample_data)

# Save sample data to CSV
sample_data.to_csv("data/sample_data.csv", index=False)
