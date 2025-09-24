import streamlit as st
import numpy as np
import joblib
import librosa

# -------------------- Load Model --------------------
model = joblib.load("voice_classifier.pkl")  # make sure this is your trained model

# -------------------- Page Config --------------------
st.set_page_config(page_title="🎙️ Voice Gender Classification", layout="wide")

# -------------------- Sidebar Navigation --------------------
menu = st.sidebar.radio("📌 Navigate", ["Home", "About"])

if menu == "Home":
    st.title("🎶 Voice Gender Classification")
    st.markdown("Record your voice and let the AI predict whether it is Male or Female 🎤")
    # All code for Home page here, properly indented

elif menu == "About":
    st.title("ℹ️ About This Project")
    st.markdown("""
        This project uses a Machine Learning model to predict voice gender.
        Developed by Muhammad Bin Ali
    """)

# -------------------- Home Page --------------------
if menu == "Home":
    st.title("🎶 Voice Gender Classification")
    st.markdown(
        "Upload a voice recording and let the AI predict whether it is **Male** or **Female** 🎤"
    )

    uploaded_file = st.file_uploader("📂 Upload your audio file (.wav or .mp3)", type=["wav", "mp3"])

    if uploaded_file is not None:
        st.audio(uploaded_file)  # Play uploaded audio

        # -------------------- Feature Extraction --------------------
        st.info("🔍 Extracting features from audio...")

        # Load audio using librosa
        y, sr = librosa.load(uploaded_file, sr=None)
        
        # Example feature extraction (replace with your actual pipeline)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        features = np.mean(mfccs.T, axis=0).reshape(1, -1)  # shape (1, 20)

        st.success("✅ Features extracted!")

        # -------------------- Prediction --------------------
       # Prediction with error handling
try:
    prediction = model.predict(features)[0]
    st.subheader("🔮 Prediction Result")
    st.write(f"**This voice is classified as:** 🎤 {prediction}")
except Exception as e:
    st.error(f"Prediction failed: {e}")

# -------------------- About Page --------------------
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


