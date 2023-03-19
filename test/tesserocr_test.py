import tesserocr
from tesserocr import PyTessBaseAPI, PSM, OEM
from tesserocr import get_languages
from PIL import Image
import numpy as np
import cv2

# api = PyTessBaseAPI()

# image = Image.open('/home/jrebernik/Magistrska/LPR-Software/dataset/images_resized/IMG20230220132545.jpg')
# print(tesserocr.image_to_text(image))

with PyTessBaseAPI(psm=PSM.OSD_ONLY, oem=OEM.LSTM_ONLY) as api:
    api.SetImageFile('/home/jrebernik/Magistrska/LPR-Software/dataset/images_resized/IMG20230220132545.jpg')
    ocr = api.GetUTF8Text()
    print(ocr)