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
colors = {
    0: (0, 255, 0),   # Class 0: Green
    1: (255, 0, 0),   # Class 1: Blue
    2: (0, 0, 255),   # Class 2: Red
    3: (255, 255, 0), # Class 3: Cyan
    4: (255, 0, 255), # Class 4: Magenta
    5: (0, 255, 255), # Class 5: Yellow
}

if data is not None:
    file_path = save_upload(data)
    image = cv2.imread(file_path)

    if image is not None:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resize = cv2.resize(image_rgb, (640, 640))
        result = model(image_resize)

        detect_label = set()

        for box in result[0].boxes:
            confidence = box.conf[0]
            if box.conf >= 0.5:
                x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                color = colors.get(class_id, (255, 255, 255))

                label = model.names[class_id] if class_id in model.names else "unknown"
                detect_label.add(label)

                # Draw image with rectangel
                cv2.rectangle(image_resize, 
                              (x_min, y_min), 
                              (x_max, y_max), 
                              color,
                              2)
                label_text = f"{label} conf:{confidence:.2f}"

                (text_width, text_height), baseline = cv2.getTextSize(label_text, 
                                                                      cv2.FONT_HERSHEY_SIMPLEX, 
                                                                      0.5, 
                                                                      2)

                cv2.rectangle(image_resize,
                              (x_min, y_min + text_height),
                              (x_min + text_width, y_min),
                               color,
                               thickness=cv2.FILLED)
                cv2.putText(image_resize, label_text, (x_min, y_min + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        st.image(image_resize, caption="Detected Objects")

        if detect_label is not None:
            for label in detect_label:
                st.markdown(f"label: {label}")
        else:
            st.error("Sorry not detection diases")


    else:
        st.error("Error reading the uploaded image. Please try again with a valid image file.")
else:
    st.info("Please upload an image file.")
        








