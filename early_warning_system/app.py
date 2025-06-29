import streamlit as st
import pandas as pd
import pickle
import joblib 

# Load model
model = pickle.load(open("model.pkl", "rb"))

st.set_page_config(page_title="Early Warning System", layout="centered")

st.title("📊 Early Warning System for At-Risk Students")
st.write("Upload student performance data to predict academic risk levels.")

# Upload file
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📁 Uploaded Data Preview")
    st.dataframe(df)

    predictions = model.predict(df)
    df["Predicted Risk Level"] = predictions

    # Color styling
    def color_risk(val):
        if val == "High":
            return "background-color: #FFCCCC; color: red"
        elif val == "Medium":
            return "background-color: #FFF3CD; color: orange"
        return "background-color: #D4EDDA; color: green"

    styled = df.style.applymap(color_risk, subset=["Predicted Risk Level"])

    st.subheader("✅ Prediction Results")
    st.dataframe(styled)

    # Download option
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Results", csv, "risk_predictions.csv", "text/csv")
else:
    st.info("Please upload a CSV file to begin.")
