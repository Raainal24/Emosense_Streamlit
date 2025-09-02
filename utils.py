import os
import tensorflow as tf
import psutil

def optimize_tensorflow():
    """Configure TensorFlow for optimal CPU performance."""
    # Suppress TensorFlow warnings and info messages
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # CPU optimization settings
    os.environ['OMP_NUM_THREADS'] = '2'
    os.environ['TF_NUM_INTEROP_THREADS'] = '2'
    os.environ['TF_NUM_INTRAOP_THREADS'] = '2'
    
    # Configure TensorFlow threading
    try:
        tf.config.threading.set_inter_op_parallelism_threads(2)
        tf.config.threading.set_intra_op_parallelism_threads(2)
    except RuntimeError:
        # TensorFlow already initialized
        pass

def get_memory_usage():
    """Get current memory usage in MB."""
    try:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # Convert to MB
    except:
        return 0

def validate_model_files():
    """Check if required model files exist."""
    required_files = [
        "emotion_detection_final_acc75.h5",
        "haarcascade_frontalface_default.xml"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    return missing_files

def get_opencv_info():
    """Get OpenCV build information."""
    import cv2
    return {
        'version': cv2.__version__,
        'build_info': cv2.getBuildInformation()
    }
