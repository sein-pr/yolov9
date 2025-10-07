from ultralytics import YOLO
import streamlit as st
import cv2

# Load your exported ONNX model
model = YOLO('best (7).pt', task='detect')

st.title('YOLO ONNX Live Inference')
run = st.checkbox('Run Inference')
FRAME_WINDOW = st.image([])

cap = cv2.VideoCapture(0)  # Use 0 for webcam

while run:
    ret, frame = cap.read()
    if not ret:
        st.write("Failed to capture image from camera.")
        break
    
    # Run inference on the frame
    results = model(frame)
    annotated_frame = results[0].plot()
    
    # Display the annotated frame
    FRAME_WINDOW.image(annotated_frame, channels="BGR")

cap.release()
st.write('Inference stopped.')