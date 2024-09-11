import os
import cv2
import streamlit as st
from utilities import save_upload
from ultralytics import YOLO

st.set_page_config(page_title="Detection object with YOLOv8")
st.sidebar.header("Detection object with YOLOv8")
st.markdown("# Detection object with YOLOv8")
model = YOLO("model/best_obb_v8.pt")
data = st.file_uploader(label="choose picture", type=["png","jpg"])

if data is not None:
    file_path = save_upload(data)
    image = cv2.imread(file_path)

    if image is not None:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_size = cv2.resize(image_rgb, (640, 640))
        result = model(image_rgb)
        for r in result:
            if r[0].boxes.conf >= 0.5:
                st.image(r.plot(), caption='Detected Objects')
            else:
                st.error("Sorry not detection diases")
    else:
        st.error("Error reading the uploaded image. Please try again with a valid image file.")
else:
    st.info("Please upload an image file.")
        








