# -*- coding: utf-8 -*-
"""
train_model.py: Script to train a simple ML model for student risk prediction.
Place 'data/student_data.csv' in your project root (or 'data/' subfolder) and this script in the project directory.
After running this script, the trained model will be saved as 'model.pkl' in the project root (or specify a 'models/' directory).
"""

# Step 1: Load and preprocess data
import pandas as pd

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
    elif score < 70:
        return 'Medium'
    else:
        return 'Low'

# Apply the risk categorization to create a 'risk' column
df['risk'] = df['final_score'].apply(categorize_risk)

# Step 3: Prepare features and labels
X = df[['final_score']]  # Features for the model (use more columns if available)
y = df['risk']

# Encode risk labels as numeric values (needed for many classifiers)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)
# (Now y_encoded contains numeric labels corresponding to 'High','Medium','Low')

# Step 4: Split the data into training and testing sets
from sklearn.model_selection import train_test_split  # splitting dataset into train and test sets:contentReference[oaicite:0]{index=0}
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Step 5: Define and train a simple model (Logistic Regression)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state=42)  # instantiate the logistic regression model
model.fit(X_train, y_train)  # Train the model on the training data:contentReference[oaicite:1]{index=1}

# (Optional) Step 6: Evaluate the model (accuracy, etc.) – not required for saving the model
# from sklearn.metrics import accuracy_score
# y_pred = model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Test accuracy: {accuracy:.2f}")

# Step 7: Save the trained model to a file
import joblib
joblib.dump(model, 'model.pkl')  # Save model to 'model.pkl':contentReference[oaicite:2]{index=2}

# The trained model 'model.pkl' can now be loaded in a Streamlit app for making predictions.
