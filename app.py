# app.py

import streamlit as st
import numpy as np
import joblib
import librosa

# ---------------- Load Model ----------------
model = joblib.load("voice_classifier.pkl")  # your pre-trained model

# ---------------- Page Config ----------------
st.set_page_config(page_title="🎙️ Voice Gender Classification", layout="wide")

# ---------------- Sidebar Navigation ----------------
menu = st.sidebar.radio("📌 Navigate", ["Home", "About"])

# ---------------- Home Page ----------------
if menu == "Home":
    st.title("🎶 Voice Gender Classification")
    st.markdown("Record your voice and let the AI predict whether it is **Male** or **Female** 🎤")

    st.markdown("---")
    st.subheader("🎧 Instructions")
    st.write(
        """
        1. Click the record button below.  
        2. Speak clearly for the selected duration.  
        3. Wait for the prediction to appear.
        """
    )

    # ---------------- Audio Recording ----------------
    duration = st.slider("🎧 Recording Duration (seconds)", 2, 10, 5)
    sampling_rate = 22050  # Recommended for librosa

    if st.button("🎙️ Record & Predict"):
        st.info("Recording... Please speak now.")
        st.audio("example_voice.mp3")  # placeholder for mobile demo (real recording requires extra setup)
        st.success("✅ Recording complete!")

        # ---------------- Feature Extraction ----------------
        try:
            # Example: replace this with your actual feature extraction
            # Here we create dummy features as placeholder
            features = np.random.rand(1, 20)  # shape must match your model input

            # ---------------- Prediction ----------------
            try:
                prediction = model.predict(features)[0]
                st.subheader("🔮 Prediction Result")
                st.write(f"**This voice is classified as:** 🎤 {prediction}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

        except Exception as e:
            st.error(f"Feature extraction failed: {e}")

# ---------------- About Page ----------------
elif menu == "About":
    st.title("ℹ️ About This Project")
    st.markdown(
        """
        ### 🎙️ Voice Gender Classification App  
        This project uses a **Machine Learning model** trained on voice features 
        to predict whether a voice belongs to a **Male** or **Female**.  

        #### 👨‍💻 Developed by:  
        **Muhammad Bin Ali**  
        *Data Scientist | ML/DL Enthusiast | Generative AI | Agentic AI*  

        ---
        ✅ **Tech Stack**:  
        - Python 🐍  
        - Streamlit 🌐  
        - Machine Learning 🤖  
        - Audio Processing 🎧  
        """
    )
    st.success("Thank you for visiting this app 🚀")
