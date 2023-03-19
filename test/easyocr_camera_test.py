import os
import sys
import time
import random
from tqdm import tqdm
import pytesseract
import tensorflow as tf
import cv2
import numpy as np
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
import easyocr

IMG_PATH = 'dataset/images/train/'
IMG_SIZE = (1080, 720)

LABELS = {
    1: {
        'id': 1,
        'name': 'plate'
    }
}

PLATE_DETECTION_THRESHOLD = 0.8
TEXT_DETECTION_THRESHOLD = 0.5
MIN_AREA_PLATE = 0.05
PLATE_CHARS = 'ABCDEFGHIJKLMNOPRSTUVZYXQ1234567890-' # All possible chars in a plate
PLATE_CITIES = ['KP', 'LJ', 'KR', 'GO', 'PO', 'NM', 'MB', 'SG', 'KK', 'MS', 'CE']



@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

def ocrTextFilter(ocr_input, threshold=0.7, min_height=45):
    """ Filter output from OCR that doesn't have confidence above threshold.
    Args:   ocr_input: raw output from EasyOCR
            threshold: threshold of detected text to keep
            min_height: minimum text height in pixels 
    Return: A single string with license plate. Return None if filtering failed. """

    plate = []

    for detection in ocr_input:
        y0 = detection[0][0][1]
        y3 = detection[0][3][1]
        score = detection[2]
        text = detection[1]
        if (y3-y0) > min_height:    # Check height only between points (0,3) because we have a paralelogram. Checking between (1,2) isn't needed.
            if score > threshold: plate.append(text)

    if len(plate) != 2:
        return None

    if plate[0] not in PLATE_CITIES: # Invalid city was detected. All plates should have one of the PLATE_CITIES
        return None
    
    if '-' in plate[1]:
        plate[1] = plate[1].replace('-', '') # Remove '-' from string

    return ''.join(plate)

def expandRoi(roi, num_pixels, img_width, img_height):
    """ Expand ROI by num_pixels. 
    args:   img_width: we must not expand more than the original image resolution
            img_height: we must not expand more than the original image resolution.
    """
    if roi[0] >= num_pixels: roi[0] -= num_pixels
    if roi[2]+num_pixels < img_height: roi[2] += num_pixels
    if roi[1] >= num_pixels: roi[1] -= num_pixels
    if roi[3]+num_pixels < img_width: roi[3]+= num_pixels
    return roi

if __name__ == '__main__':

    # Load pipeline config and build a detection model
    configs = config_util.get_configs_from_pipeline_file('tensorflow/workspace/models/plate_model_320/pipeline.config')
    detection_model = model_builder.build(model_config=configs['model'], is_training=False)

    # Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join('tensorflow/workspace/models/plate_model_320/check', 'ckpt-3')).expect_partial() # Use latest checkpoint generated

    # Configure EasyOCR reader
    reader = easyocr.Reader(['en'], gpu=True)

    # cap = cv2.VideoCapture('dataset/collected_images/VID20230305180320.mp4')
    cap = cv2.VideoCapture(0)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 100)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 80)
    while cap.isOpened():
        # Load the image using OpenCV
        _, img = cap.read()
        img = cv2.resize(img, IMG_SIZE)
        img_np = np.array(img) # Convert image to numpy array for faster processing
        # cv2.imshow('Loaded img', img_np)
        
        input_tensor = tf.convert_to_tensor(np.expand_dims(img_np, 0), dtype=tf.float32) # Convert img to tensor
        detections = detect_fn(input_tensor)
        
        # Convert TensorObject structure to numpy arrays
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                    for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # Detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        image_np_with_detections = img_np.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
                    image_np_with_detections,
                    detections['detection_boxes'],
                    detections['detection_classes']+1, # We need an offset of 1, else it doesnt work
                    detections['detection_scores'],
                    LABELS,
                    use_normalized_coordinates=True,
                    max_boxes_to_draw=1, # Draw only one plate (the one with highest probability)
                    min_score_thresh=PLATE_DETECTION_THRESHOLD,
                    agnostic_mode=False)
        
        # cv2.imshow('Object detection',  image_np_with_detections)

        if detections['detection_scores'][0] > PLATE_DETECTION_THRESHOLD: # Boxes are sorted from most probable to least probable. So just take the 0th index and we got our most probable detection
            roi = detections['detection_boxes'][0]
            
            roi_area = (roi[2]-roi[0]) * (roi[3]-roi[1])
            print('ROI area:', roi_area)

            if roi_area > MIN_AREA_PLATE:
                roi *= [IMG_SIZE[1], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[0]] # detections has normalised boxes. So we multiply by image dimentions
                roi = roi.astype(np.int64)
                roi = expandRoi(roi, 10, IMG_SIZE[0], IMG_SIZE[1]) # Expand ROI by 10 pixels

                # print(roi) # Print Region Of Interest

                img_roi = cv2.resize(img_np[roi[0]:roi[2], roi[1]:roi[3]], (205,50)) # Crop image to ROI and resize

                ocr_result = reader.readtext(img_roi, allowlist=PLATE_CHARS, min_size=60, height_ths=0.1, width_ths=0.1) # Perform OCR
                # ocr_result = pytesseract.image_to_string(img_roi, config='--psm 11')
                # ocr_result = pytesseract.image_to_data(img_roi, config='--psm 11', nice=0)
                detected_plate = ocrTextFilter(ocr_result, threshold=TEXT_DETECTION_THRESHOLD)
                
                print(ocr_result)
                print(detected_plate)

                for box in ocr_result:
                    start_point = (int(box[0][0][0]), int(box[0][0][1]))
                    end_point = (int(box[0][2][0]), int(box[0][2][1]))
                    cv2.rectangle(img_roi, start_point, end_point, (255,255,255), 2)

                cv2.imshow('Region of interest', img_roi)

        cv2.imshow('Object detection',  image_np_with_detections)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()


