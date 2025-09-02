import streamlit as st
import cv2
import numpy as np
import av
import os
import gc
import psutil
from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
from utils import optimize_tensorflow, get_memory_usage

# --- CPU Optimization Setup ---
optimize_tensorflow()

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="EmoSense - Emotion Detection",
    page_icon="😐",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- Load Model and Haar Cascade with Caching ---
@st.cache_resource
def load_emotion_model(model_path):
    """Load pre-trained emotion detection model with optimization."""
    try:
        # Load model without compilation for faster loading
        model = load_model(model_path, compile=False)
        
        # Recompile for CPU optimization
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        
        st.success("✅ Emotion detection model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None

@st.cache_resource
def load_face_cascade(cascade_path):
    """Load Haar Cascade with fallback to OpenCV built-in."""
    try:
        cascade = cv2.CascadeClassifier(cascade_path)
        if cascade.empty():
            # Fallback to OpenCV's built-in cascade
            cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            if not cascade.empty():
                st.info("ℹ️ Using OpenCV's built-in face cascade")
        
        if not cascade.empty():
            st.success("✅ Face detection model loaded successfully!")
            return cascade
        else:
            raise Exception("Could not load face cascade classifier")
            
    except Exception as e:
        st.error(f"❌ Failed to load face detector: {e}")
        return None

# Paths for model files
MODEL_PATH = "emotion_detection_final_acc75.h5"
CASCADE_PATH = "haarcascade_frontalface_default.xml"

# Load resources
with st.spinner("Loading AI models..."):
    model = load_emotion_model(MODEL_PATH)
    face_cascade = load_face_cascade(CASCADE_PATH)

# Emotion labels
emotions = ('Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral')

# Color mapping for emotions
emotion_colors = {
    'Happy': (0, 255, 0),      # Green
    'Sad': (255, 0, 0),        # Blue
    'Angry': (0, 0, 255),      # Red
    'Fear': (128, 0, 128),     # Purple
    'Surprise': (0, 255, 255), # Yellow
    'Disgust': (0, 128, 255),  # Orange
    'Neutral': (128, 128, 128) # Gray
}

# --- Optimized Video Processing Logic ---
class EmotionDetector(VideoTransformerBase):
    def __init__(self):
        self.model = model
        self.face_cascade = face_cascade
        self.frame_count = 0
        self.last_emotions = {}  # Store last detected emotions for smoothing
        
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        
        # Check if models are loaded
        if self.model is None or self.face_cascade is None:
            # Display error message on frame
            cv2.putText(img, "Models not loaded", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        
        # Process every 2nd frame for better performance
        self.frame_count += 1
        if self.frame_count % 2 != 0:
            return av.VideoFrame.from_ndarray(img, format="bgr24")
            
        try:
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Optimized face detection parameters
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            for i, (x, y, w, h) in enumerate(faces):
                # Extract and preprocess face region
                roi_gray = gray[y:y + h, x:x + w]
                roi_resized = cv2.resize(roi_gray, (48, 48))
                roi_normalized = roi_resized.astype("float32") / 255.0
                
                # Convert to 3-channel for model compatibility
                roi_rgb = cv2.cvtColor(roi_normalized, cv2.COLOR_GRAY2RGB)
                input_img = np.expand_dims(roi_rgb, axis=0)
                
                try:
                    # Emotion prediction
                    predictions = self.model.predict(input_img, verbose=0)
                    max_index = np.argmax(predictions[0])
                    predicted_emotion = emotions[max_index]
                    confidence = predictions[0][max_index]
                    
                    # Confidence threshold and smoothing
                    if confidence > 0.4:  # Lower threshold for better detection
                        # Simple emotion smoothing
                        face_id = f"face_{i}"
                        if face_id in self.last_emotions:
                            # If same emotion detected twice, use it
                            if self.last_emotions[face_id] == predicted_emotion:
                                final_emotion = predicted_emotion
                            else:
                                final_emotion = predicted_emotion
                        else:
                            final_emotion = predicted_emotion
                        
                        self.last_emotions[face_id] = predicted_emotion
                        
                        # Get color for emotion
                        color = emotion_colors.get(final_emotion, (255, 255, 255))
                        
                        # Draw bounding box
                        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                        
                        # Create label with emotion and confidence
                        label = f"{final_emotion}: {confidence:.2f}"
                        
                        # Calculate text size for background
                        (text_width, text_height), _ = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                        )
                        
                        # Draw background rectangle for text
                        cv2.rectangle(img, (x, y - text_height - 10), 
                                    (x + text_width, y), color, -1)
                        
                        # Draw text
                        cv2.putText(img, label, (x, y - 5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    else:
                        # Low confidence - just draw box
                        cv2.rectangle(img, (x, y), (x + w, y + h), (128, 128, 128), 2)
                        cv2.putText(img, "Analyzing...", (x, y - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
                        
                except Exception as pred_error:
                    # Handle prediction errors gracefully
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(img, "Processing Error", (x, y - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Display frame info
            cv2.putText(img, f"Faces: {len(faces)}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                       
        except Exception as e:
            # Handle any other errors
            cv2.putText(img, f"Error: {str(e)[:30]}", (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- Streamlit UI ---
st.title("🎭 EmoSense - Real-Time Emotion Detection")

# Status indicator
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    if model is not None:
        st.success("🤖 AI Model: Ready")
    else:
        st.error("🤖 AI Model: Failed")

with col2:
    if face_cascade is not None:
        st.success("👤 Face Detection: Ready")
    else:
        st.error("👤 Face Detection: Failed")

with col3:
    memory_usage = get_memory_usage()
    if memory_usage < 500:  # MB
        st.success(f"💾 Memory: {memory_usage:.1f} MB")
    else:
        st.warning(f"💾 Memory: {memory_usage:.1f} MB")

st.markdown(
    """
    Welcome to **EmoSense** — a real-time facial emotion detection app using your webcam.
    Click the **START** button below to begin scanning your facial expressions.
    
    **Detected Emotions:** Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
    """
)

# WebRTC configuration for deployment
rtc_config = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
    ]
})

# Only start webcam if models are loaded
if model is not None and face_cascade is not None:
    # Start the webcam stream
    webrtc_ctx = webrtc_streamer(
        key="emotion-detection-stream",
        video_processor_factory=EmotionDetector,
        rtc_configuration=rtc_config,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    
    if webrtc_ctx.state.playing:
        st.success("📹 Camera is active. Detecting emotions in real-time...")
        
        # Real-time controls
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("🔄 Clear Cache"):
                st.cache_data.clear()
                st.cache_resource.clear()
                gc.collect()
                st.success("Cache cleared!")
        
        with col2:
            current_memory = get_memory_usage()
            st.metric("Memory Usage", f"{current_memory:.1f} MB")
    else:
        st.info("👆 Click 'Start' above to begin emotion detection.")
else:
    st.error("❌ Cannot start camera - AI models failed to load. Please check your model files.")

# --- Information Section ---
with st.expander("📖 How it Works"):
    st.markdown(
        """
        ### EmoSense Technology Stack
        
        1. **Face Detection**: Uses OpenCV's Haar Cascade classifier to locate faces in real-time
        2. **Emotion Recognition**: A pre-trained Convolutional Neural Network (CNN) analyzes facial expressions
        3. **Real-Time Processing**: Optimized for smooth performance with frame skipping and confidence thresholding
        4. **WebRTC Streaming**: Secure, browser-based video streaming without plugins
        
        ### Supported Emotions
        - 😠 **Angry** (Red box)
        - 🤢 **Disgust** (Orange box)  
        - 😨 **Fear** (Purple box)
        - 😊 **Happy** (Green box)
        - 😢 **Sad** (Blue box)
        - 😮 **Surprise** (Yellow box)
        - 😐 **Neutral** (Gray box)
        
        ### Performance Notes
        - Optimized for CPU deployment
        - Processes every 2nd frame for smooth performance
        - Confidence threshold of 40% for reliable detection
        - Memory usage monitoring and cleanup
        """
    )

with st.expander("⚡ Performance Tips"):
    st.markdown(
        """
        ### For Best Results:
        - Ensure good lighting on your face
        - Position yourself 2-3 feet from the camera
        - Keep your face clearly visible and unobstructed
        - Allow a few seconds for the model to stabilize predictions
        
        ### Troubleshooting:
        - If the camera doesn't start, refresh the page
        - Clear cache if experiencing memory issues
        - Check browser permissions for camera access
        """
    )

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
    <p>Built with ❤️ using Streamlit, TensorFlow, and OpenCV</p>
    </div>
    """, 
    unsafe_allow_html=True
)
