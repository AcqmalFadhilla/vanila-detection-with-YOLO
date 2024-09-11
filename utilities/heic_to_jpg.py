import os
from PIL import Image
import pillow_heif


ROOT = "Dataset"
categories = ["Akar Busuk", "Buah Busuk", "Daun Berpenyakit"]
data = {category: [] for category in categories}

for category in categories:
    category_dir = os.path.join(ROOT, category)
    if os.path.isdir(category_dir):
        data[category].extend(
            os.path.join(category_dir, filename) 
            for filename in os.listdir(category_dir))

def convert_heic_to_jpg(heic_path, jpg_path):
    heif_file = pillow_heif.read_heif(heic_path)
    image = Image.frombytes(heif_file.mode,
                             heif_file.size, 
                             heif_file.data, 
                             "raw", 
                             heif_file.mode, 
                             heif_file.stride)
    
    
    image.save(jpg_path, "JPEG")

for category in data:
    for i, file in enumerate(data[category]):
        jpg_file = os.path.splitext(file)[0]
        convert_heic_to_jpg(file, f"{jpg_file}.jpg")