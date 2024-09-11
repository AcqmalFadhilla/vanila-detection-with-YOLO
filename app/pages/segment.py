import streamlit as st
import cv2
from utilities import save_upload
from ultralytics import YOLO


st.set_page_config(page_title="Detection Segmentation with YOLOv9")
st.sidebar.header("Detection Segmentation with YOLOv9")
st.markdown("# Detection Segmentation with YOLOv9")
model = YOLO("model/best_segment_v9.pt")
data = st.file_uploader("chosee picture", type=["png", "jpg"])

if data is not None:
    file_path = save_upload(data)
    image = cv2.imread(file_path)
    
    if image is not None:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resize = cv2.resize(image_rgb, (640, 640))
        results = model(image_rgb)
        for r in results:
            # detected_any = False
            # print(r.boxes.conf)
            if r[0].boxes.conf >= 0.5:
                st.image(r.plot(), caption='Detected Objects')
            # if len(r.boxes) > 0 and r.boxes.conf.max() >= 0.5:
            #         detected_any = True
            #         st.image(r.plot(), caption='Detected Objects')
            else:
                st.error("Sorry not detection diases")
    else:
        st.error("Error reading the uploaded image. Please try again with a valid image file.")
else:
    st.info("Please upload an image file.")