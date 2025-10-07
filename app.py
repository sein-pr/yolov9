from ultralytics import YOLO
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time

model = YOLO('best (3).onnx', task='detect')

st.title('YOLO ONNX Live Inference')

run = st.checkbox('Run Inference')
FRAME_WINDOW = st.image([])

if run:
    camera = st.camera_input("Camera", key=f"camera_{time.time()}")
    if camera:
        image = Image.open(camera)
        frame = np.array(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        results = model(frame)
        annotated_frame = results[0].plot()
        
        FRAME_WINDOW.image(annotated_frame, channels="BGR")
        st.rerun()  # Refresh for next frame