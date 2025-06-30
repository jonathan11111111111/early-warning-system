import pandas as pd
import pickle
import streamlit as st

# Load model
try:
    model = pickle.load(open("data/model.pkl", "rb"))
except FileNotFoundError:
    st.error("❌ Model file not found! Make sure 'data/model.pkl' exists.")
    st.stop()

# Page config
st.set_page_config(page_title="📊 Early Warning System", layout="centered")

st.title("📊 Early Warning System for At-Risk Students")
st.write("Upload student performance data to predict academic risk levels (High, Medium, Low).")

# Upload file
uploaded_file = st.file_uploader("📁 Upload a CSV file", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        st.subheader("🔍 Uploaded Data Preview")
        st.dataframe(df)

        # Clean the data: remove rows with invalid values (e.g., '...')
        df_cleaned = df.replace('...', pd.NA).dropna()

        # Convert all numeric columns (if possible)
        for col in df_cleaned.columns:
            df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')

        df_cleaned = df_cleaned.dropna()  # Drop rows where conversion failed

        if df_cleaned.empty:
            st.warning("⚠️ All rows were dropped due to invalid or missing data.")
        else:
            # Predict
            predictions = model.predict(df_cleaned)
            df_cleaned["Predicted Risk Level"] = predictions

            # Color styling
            def color_risk(val):
                if val == "High":
                    return "background-color: #FFCCCC; color: red"
                elif val == "Medium":
                    return "background-color: #FFF3CD; color: orange"
                elif val == "Low":
                    return "background-color: #D4EDDA; color: green"
                return ""

            st.subheader("✅ Prediction Results")
            st.dataframe(df_cleaned.style.applymap(color_risk, subset=["Predicted Risk Level"]))

            # Download option
            csv = df_cleaned.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Results", csv, "risk_predictions.csv", "text/csv")

    except Exception as e:
        st.error(f"❌ Error reading the uploaded file: {e}")
else:
    st.info("📎 Please upload a CSV file to begin.")
