import streamlit as st
import numpy as np
import joblib
import sounddevice as sd
import wavio

# -------------------- Load Model --------------------
model = joblib.load("voice_classifier.pkl")  # make sure it's saved correctly

# -------------------- Page Config --------------------
st.set_page_config(page_title="🎙️ Voice Gender Classification", layout="wide")

# -------------------- Sidebar Navigation --------------------
menu = st.sidebar.radio("📌 Navigate", ["Home", "About"])

# -------------------- Home Page --------------------
if menu == "Home":
    st.title("🎶 Voice Gender Classification")
    st.markdown("Record your voice and let the AI predict whether it is **Male** or **Female** 🎤")

    duration = st.slider("🎧 Recording Duration (seconds)", 2, 10, 5)
    samplerate = 44100  # CD quality

    if st.button("🎙️ Record Voice"):
        st.info("Recording... Speak now!")
        recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype="float32")
        sd.wait()
        wavio.write("recorded.wav", recording, samplerate, sampwidth=2)
        st.success("✅ Recording complete! Saved as `recorded.wav`")

        # 👉 Dummy Example: extract features
        # Replace this with your actual feature extraction pipeline
        features = np.random.rand(1, 20)  # Example only

        # Prediction
        prediction = model.predict(features)[0]
        st.subheader("🔮 Prediction Result")
        st.write(f"**This voice is classified as:** 🎤 {prediction}")

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
        *Data Scientist | AI/ML Engineer | Generative AI | Agentic AI*  

        ---
        ✅ **Tech Stack**:  
        - Python 🐍  
        - Streamlit 🌐  
        - Machine Learning 🤖  
        - Audio Processing 🎧  
        """
    )
    st.success("Thank you for visiting this app 🚀")
