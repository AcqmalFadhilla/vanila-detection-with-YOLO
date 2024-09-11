import tempfile
import os

def save_upload(file_upload):
    dir = tempfile.mkdtemp()
    file_dir = os.path.join(dir, file_upload.name)
    with open(file_dir, "wb") as f:
        f.write(file_upload.getbuffer())
    return file_dir

