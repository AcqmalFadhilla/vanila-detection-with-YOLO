import os
import cv2

ROOT = "Dataset/image"
images = []

for file in os.listdir(ROOT):
    images.append(os.path.join(ROOT, file))

for idx, image in enumerate(images):
    print(image)
    img = cv2.imread(image)
    image_resize = cv2.resize(img, (600, 600))
    cv2.imwrite(f"Dataset/resize_image/{idx}.png", image_resize)

