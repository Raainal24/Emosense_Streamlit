import streamlit as st
import cv2
import numpy as np
import av
from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="EmoSense - Emotion Detection",
    page_icon="😐",
    layout="centered"
)

# --- Load Model and Haar Cascade with Caching ---
@st.cache_resource
def load_emotion_model(model_path):
    """Load pre-trained emotion detection model."""
    try:
        return load_model(model_path)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

@st.cache_resource
def load_face_cascade(cascade_path):
    """Load Haar Cascade classifier for face detection."""
    try:
        return cv2.CascadeClassifier(cascade_path)
    except Exception as e:
        st.error(f"Failed to load face detector: {e}")
        return None

# Paths (make sure files are in the same directory as app.py)
MODEL_PATH = "emotion_detection_final_acc75.h5"
CASCADE_PATH = "haarcascade_frontalface_default.xml"

# Load resources
model = load_emotion_model(MODEL_PATH)
face_cascade = load_face_cascade(CASCADE_PATH)
emotions = ('Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral')

# --- Video Processing Logic ---
class EmotionDetector(VideoTransformerBase):
    def __init__(self):
        self.model = model
        self.face_cascade = face_cascade

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        if self.model is None or self.face_cascade is None:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)
            resized_img = cv2.resize(roi_color, (48, 48)).astype("float32") / 255.0
            input_img = np.expand_dims(resized_img, axis=0)

            predictions = self.model.predict(input_img)
            max_index = np.argmax(predictions[0])
            predicted_emotion = emotions[max_index]

            if predicted_emotion == "Neutral":
                predicted_emotion = "Surprise"

            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                img, predicted_emotion, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
            )

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- Streamlit UI ---
st.title("🎭 EmoSense - Real-Time Emotion Detection")
st.markdown(
    """
    Welcome to **EmoSense** — a real-time facial emotion detection app using your webcam.
    Click the **START** button below to begin scanning your facial expressions.
    """
)

# WebRTC configuration (necessary for deployment on public platforms)
rtc_config = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

# Start the webcam stream
webrtc_ctx = webrtc_streamer(
    key="emotion-stream",
    video_processor_factory=EmotionDetector,
    rtc_configuration=rtc_config,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

if webrtc_ctx.state.playing:
    st.success("Camera is active. Detecting emotions in real-time...")
else:
    st.info("Click 'Start' above to begin.")

st.markdown(
    """
    ---
    ### How it Works
    1.  **Face Detection**: Uses a Haar Cascade classifier to find faces in the video feed.
    2.  **Emotion Prediction**: A pre-trained Convolutional Neural Network (CNN) analyzes each face to predict one of seven emotions.
    3.  **Real-Time Display**: The results are overlaid on the video stream and displayed back to you.
    """
)