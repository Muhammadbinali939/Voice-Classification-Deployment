import streamlit as st
import numpy as np
import joblib
import librosa
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings

# -------------------- Load Model --------------------
model = joblib.load("voice_classifier.pkl")  # Make sure your trained model is here

# -------------------- Page Config --------------------
st.set_page_config(page_title="🎙️ Voice Gender Classification", layout="wide")

# -------------------- Sidebar Navigation --------------------
menu = st.sidebar.radio("📌 Navigate", ["Home", "About"])

# -------------------- Home Page --------------------
if menu == "Home":
    st.title("🎶 Voice Gender Classification")
    st.markdown(
        """
        Record your voice directly in the browser, and let the AI predict whether it is **Male** or **Female** 🎤
        """
    )

    st.markdown("### 🎙️ Record your voice")
    st.info("Click 'Start Recording', speak, and wait a few seconds for prediction.")

    # -------------------- WebRTC Recorder --------------------
    webrtc_ctx = webrtc_streamer(
        key="voice-recorder",
        mode=WebRtcMode.SENDONLY,
        media_stream_constraints={"audio": True, "video": False},
        client_settings=ClientSettings(
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"audio": True, "video": False},
        ),
    )

    # -------------------- Prediction --------------------
    if webrtc_ctx.audio_receiver:
        audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
        if audio_frames:
            # Convert audio frames to numpy array
            audio_data = np.hstack([f.to_ndarray() for f in audio_frames])

            # -------------------- Feature Extraction --------------------
            # Example: MFCC extraction (replace with your actual pipeline if needed)
            try:
                mfccs = librosa.feature.mfcc(y=audio_data.astype(float), sr=44100, n_mfcc=20)
                features = np.mean(mfccs.T, axis=0).reshape(1, -1)

                # Prediction
                prediction = model.predict(features)[0]

                st.subheader("🔮 Prediction Result")
                st.write(f"**This voice is classified as:** 🎤 {prediction}")
            except Exception as e:
                st.error(f"Error during feature extraction or prediction: {e}")

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
