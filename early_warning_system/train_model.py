import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the dataset
# Ensure the file path is correct and matches your dataset location
df = pd.read_csv("data/student_data.csv")

# Remove rows with missing values in the 'final_score' column
df = df.dropna(subset=["final_score"])

# Define a function to categorize risk levels
def categorize_risk(score):
    """
    Categorize a numeric score into a risk level:
    - 'High' for score < 50
    - 'Medium' for score 50–69
    - 'Low' for score >= 70
    """
    if score < 50:
        return "High"
    elif 50 <= score <= 69:
        return "Medium"
    else:
        return "Low"

# Apply the function to create a new column for risk levels
df["risk_level"] = df["final_score"].apply(categorize_risk)

# Prepare features (X) and target (y)
X = df[["attendance", "homework_score", "midterm_score", "final_score"]]  # Features
y = df["risk_level"]  # Target

# Convert the target variable to numeric values
y = y.map({"High": 0, "Medium": 1, "Low": 2})

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForestClassifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save the trained model to a file
with open("data/model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model training complete. The model has been saved as 'data/model.pkl'.")