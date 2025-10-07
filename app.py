from ultralytics import YOLO
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time

model = YOLO("best (3).onnx")

st.title("📹 YOLO Live Inference (Web-Friendly)")

frame_placeholder = st.empty()

run = st.checkbox("Run live detection")

while run:
    img_file = st.camera_input("Camera feed", key=time.time())

    if img_file is not None:
        image = Image.open(img_file)
        frame = np.array(image)

        results = model(frame)
        annotated_frame = results[0].plot()

        frame_placeholder.image(annotated_frame, channels="BGR", use_column_width=True)
    else:
        st.warning("Waiting for camera input...")

    # Control refresh rate
    time.sleep(1)
