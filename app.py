import streamlit as st
import cv2
import numpy as np
import time
from estimator import ImprovedGlucoseEstimator
import matplotlib.pyplot as plt
import io
from PIL import Image
import tempfile
import base64
from threading import Thread
import queue

# Initialize the model
@st.cache_resource
def get_estimator():
    return ImprovedGlucoseEstimator()

# Function to convert OpenCV image to format for display
def convert_to_image(frame):
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

# Function to create a matplotlib figure for glucose trends
def create_glucose_plot(time_values, glucose_values):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time_values, glucose_values, 'b-')
    ax.set_ylim(70, 200)
    ax.set_title('Estimated Glucose Level')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Glucose (mg/dL)')
    ax.grid(True)
    
    # Convert to image
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)
    return buf

# Set up the Streamlit app
st.set_page_config(page_title="Non-Invasive Glucose Monitoring", page_icon="🩸", layout="wide")

# Title and description
st.title("Non-Invasive Glucose Monitoring")
st.markdown("""
This application uses computer vision to estimate glucose levels from eye features.
Please note this is a simulation and not a medical device - do not use for medical decisions.
""")

# Initialize session state
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'estimator' not in st.session_state:
    st.session_state.estimator = get_estimator()
if 'glucose_history' not in st.session_state:
    st.session_state.glucose_history = []
if 'time_history' not in st.session_state:
    st.session_state.time_history = []
if 'frame_queue' not in st.session_state:
    st.session_state.frame_queue = queue.Queue(maxsize=1)
if 'result_queue' not in st.session_state:
    st.session_state.result_queue = queue.Queue(maxsize=10)
if 'start_time' not in st.session_state:
    st.session_state.start_time = time.time()

# Create two columns for the layout
col1, col2 = st.columns([3, 2])

# Column 1: Camera feed and controls
with col1:
    # Camera options
    camera_option = st.selectbox(
        'Select Camera Source',
        options=["Webcam", "Upload Video"]
    )
    
    # Placeholder for the camera feed
    camera_placeholder = st.empty()
    
    # Start/Stop button
    if st.button('Start Monitoring' if not st.session_state.processing else 'Stop Monitoring'):
        st.session_state.processing = not st.session_state.processing
        
        if st.session_state.processing:
            # Reset for new session
            st.session_state.estimator.initialize_session()
            st.session_state.glucose_history = []
            st.session_state.time_history = []
            st.session_state.start_time = time.time()

# Column 2: Results and visualization
with col2:
    # Current glucose reading
    glucose_value = st.empty()
    
    # Eye tracking visualization
    eye_tracking = st.empty()
    
    # Glucose trend chart
    trend_chart = st.empty()
    
    # Disclaimer
    st.markdown("""
    **Disclaimer:** This application is for demonstration purposes only and not intended for
    medical diagnosis. The glucose values shown are simulated and may not reflect actual blood glucose.
    """)

# Function for processing frames in a background thread
def process_frames():
    while st.session_state.processing:
        try:
            if not st.session_state.frame_queue.empty():
                frame = st.session_state.frame_queue.get(timeout=1)
                
                # Process the frame with our model
                processed_frame, glucose = st.session_state.estimator.process_frame(frame)
                
                # Get eye tracking data
                eye_image = st.session_state.estimator.eye_display
                
                # Put results in queue for the main thread to display
                if glucose is not None:
                    current_time = time.time() - st.session_state.start_time
                    result = {
                        'processed_frame': processed_frame,
                        'glucose': glucose,
                        'eye_image': eye_image,
                        'time': current_time
                    }
                    # Make sure we don't block if the queue is full
                    try:
                        st.session_state.result_queue.put(result, block=False)
                    except queue.Full:
                        pass
                else:
                    # Just update the frame if no glucose reading yet
                    result = {
                        'processed_frame': processed_frame,
                        'glucose': None,
                        'eye_image': eye_image
                    }
                    try:
                        st.session_state.result_queue.put(result, block=False)
                    except queue.Full:
                        pass
        except queue.Empty:
            time.sleep(0.1)
        except Exception as e:
            st.error(f"Error in processing thread: {e}")
            st.session_state.processing = False
            break

# Main app logic
if st.session_state.processing:
    # Start background processing thread if not already running
    if not hasattr(st.session_state, 'processing_thread') or not st.session_state.processing_thread.is_alive():
        st.session_state.processing_thread = Thread(target=process_frames)
        st.session_state.processing_thread.daemon = True
        st.session_state.processing_thread.start()
    
    # Handle different camera sources
    if camera_option == "Webcam":
        # Use webcam
        cap = cv2.VideoCapture(0)
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                # Put frame in queue for processing
                try:
                    if not st.session_state.frame_queue.full():
                        st.session_state.frame_queue.put(frame, block=False)
                except queue.Full:
                    pass
                
                # Display the camera feed
                camera_placeholder.image(convert_to_image(frame), caption="Camera Feed", use_column_width=True)
            else:
                camera_placeholder.error("Failed to capture frame from webcam")
                st.session_state.processing = False
            
            cap.release()
        else:
            camera_placeholder.error("Failed to open webcam")
            st.session_state.processing = False
    
    elif camera_option == "Upload Video":
        # Allow video upload
        uploaded_file = camera_placeholder.file_uploader("Upload a video", type=['mp4', 'avi', 'mov'])
        
        if uploaded_file is not None:
            # Save uploaded file to temp location
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            
            # Process the video
            cap = cv2.VideoCapture(tfile.name)
            
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    # Put frame in queue for processing
                    try:
                        if not st.session_state.frame_queue.full():
                            st.session_state.frame_queue.put(frame, block=False)
                    except queue.Full:
                        pass
                    
                    # Display the video frame
                    camera_placeholder.image(convert_to_image(frame), caption="Video Frame", use_column_width=True)
                else:
                    camera_placeholder.error("Failed to read video frame")
                    st.session_state.processing = False
                
                cap.release()
            else:
                camera_placeholder.error("Failed to open video file")
                st.session_state.processing = False
    
    # Check for and display results
    if not st.session_state.result_queue.empty():
        try:
            result = st.session_state.result_queue.get(timeout=1)
            
            # Display processed frame
            camera_placeholder.image(convert_to_image(result['processed_frame']), caption="Processed Feed", use_column_width=True)
            
            # Display eye tracking
            if 'eye_image' in result and result['eye_image'] is not None:
                eye_tracking.image(result['eye_image'], caption="Eye Tracking", width=300)
            
            # Update glucose reading if available
            if result['glucose'] is not None:
                # Format text and color based on glucose level
                glucose = result['glucose']
                if glucose < 70:
                    color = "red"
                    status = "LOW"
                elif glucose > 140:
                    color = "orange"
                    status = "HIGH"
                else:
                    color = "green"
                    status = "NORMAL"
                
                # Display glucose value with appropriate styling
                glucose_value.markdown(f"<h1 style='color: {color};'>Glucose: {glucose:.1f} mg/dL ({status})</h1>", unsafe_allow_html=True)
                
                # Add to history
                st.session_state.glucose_history.append(glucose)
                st.session_state.time_history.append(result['time'])
                
                # Update trend chart
                if len(st.session_state.glucose_history) > 1:
                    plot_buf = create_glucose_plot(st.session_state.time_history, st.session_state.glucose_history)
                    trend_chart.image(plot_buf, caption="Glucose Trend", use_column_width=True)
        except queue.Empty:
            pass
else:
    # Display instructions when not processing
    camera_placeholder.markdown("""
    ### Instructions
    1. Choose your camera source (webcam or upload a video)
    2. Click "Start Monitoring" to begin glucose estimation
    3. Look directly at the camera, ensuring your eyes are visible
    4. The system will track your eyes and estimate glucose levels
    5. Click "Stop Monitoring" when finished
    """)
    
    # Display placeholder for glucose reading
    glucose_value.markdown("### Glucose: -- mg/dL")