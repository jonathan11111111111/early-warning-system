import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Load dataset
data = pd.read_csv("data/student_data.csv")

# Define features and label (create a 'risk_level' column if it doesn't exist)
# For demo, let's create one based on final_score
def label_risk(score):
    if score < 50:
        return "High"
    elif score < 70:
        return "Medium"
    else:
        return "Low"

data["risk_level"] = data["final_score"].apply(label_risk)

X = data[["attendance", "homework_score", "midterm_score", "final_score"]]
y = data["risk_level"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model trained and saved as model.pkl")
