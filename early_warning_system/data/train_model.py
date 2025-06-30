# -*- coding: utf-8 -*-
"""
Train a machine learning model to predict student risk levels.
"""

# Step 1: Load and preprocess data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle

# Load the CSV data (adjust path if using a subfolder like 'data/student_data.csv')
df = pd.read_csv('data/student_data.csv')

# Remove any rows with missing 'final_score' values
df = df.dropna(subset=['final_score'])

# Step 2: Create risk labels based on scores
def categorize_risk(score):
    """
    Categorize a numeric score into a risk level:
    - 'High' for score < 50
    - 'Medium' for score 50–69
    - 'Low' for score >= 70
    """
    if score < 50:
        return 'High'
    elif 50 <= score <= 69:
        return 'Medium'
    else:
        return 'Low'

# Apply the function to create a new column
df['risk_level'] = df['final_score'].apply(categorize_risk)

# Step 3: Prepare data for training
X = df.drop(columns=['final_score', 'risk_level'])  # Features
y = df['risk_level']  # Target

# Convert categorical target to numeric
y = y.map({'High': 0, 'Medium': 1, 'Low': 2})

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 5: Evaluate the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Step 6: Save the trained model
with open('data/model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model training complete. Saved as 'data/model.pkl'.")
