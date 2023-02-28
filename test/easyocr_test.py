import os
import sys
import time
import tensorflow as tf
import cv2
import numpy as np
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
import easyocr


LABELS = {
    1: {
        'id': 1,
        'name': 'plate'
    }
}

DETECTION_THRESHOLD = 0.8
PLATE_CHARS = 'ABCDEFGHIJKLMNOPRSTUVZYXQ1234567890-' # All possible chars in a plate
PLATE_CITIES = ['KP', 'LJ', 'KR', 'GO', 'PO', 'NM', 'MB', 'SG', 'KK', 'MS', 'CE']



@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

def ocrTextFilter(ocr_input, plate_cities, threshold=0.7, min_height=60):
    """ Filter output from OCR that doesn't have confidence above threshold.
    Args:   ocr_input: raw output from EasyOCR
            plate_cities: all possible cities on a plate (first item in ocr_input)
            threshold: threshold of detected text to keep
            min_height: minimum text height in pixels 
    Return: A single string with license plate. Return None if filtering failed. """

    plate = []
    for s in ocr_input:
        if s[0][2][1] - s[0][0][1] > min_height:
            if s[2] > threshold:
                plate.append(s[1])

    if len(plate) != 2:
        return None

    if plate[0] not in plate_cities: # Invalid city was detected. All plates should have one of the PLATE_CITIES
        return None

    return ' '.join(plate)


if __name__ == '__main__':

    # Limit Tensorflow GPU consumption
    # gpus = tf.config.list_physical_devices('GPU')
    # if gpus:
    #     try: 
    #         tf.config.experimental.set_virtual_device_configuration(
    #             gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3381)])
    #     except:
    #         print(sys.exc_info()[0])

    # Load pipeline config and build a detection model
    configs = config_util.get_configs_from_pipeline_file('tensorflow/workspace/models/plate_model_320/pipeline.config')
    detection_model = model_builder.build(model_config=configs['model'], is_training=False)

    # Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join('tensorflow/workspace/models/plate_model_320/check', 'ckpt-3')).expect_partial() # Use latest checkpoint generated

    reader = easyocr.Reader(['en'], gpu=False)

    # Get all .jpg images in test folder
    jpg_images = [filename for filename in os.listdir('dataset/images/test/') if filename.endswith('.jpg')]

    for image in jpg_images: # Loop through all images in test folder
        # Load the image using OpenCV
        img = cv2.imread('dataset/images/test/{}'.format(image))
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
                    min_score_thresh=DETECTION_THRESHOLD,
                    agnostic_mode=False)
        
        cv2.imshow('Object detection',  image_np_with_detections)

        width = image_np_with_detections.shape[1]
        height = image_np_with_detections.shape[0]

        if detections['detection_scores'][0] > DETECTION_THRESHOLD: # Boxes are sorted from most probable to least probable. So just take the 0th index and we got our most probable detection
            roi = detections['detection_boxes'][0] * [height, width, height, width] # detections has normalised boxes. So we multiply by image dimentions
            roi = roi.astype(np.int64)
            print(roi) # Print Region Of Interest

            img_roi = cv2.resize(img_np[roi[0]:roi[2], roi[1]:roi[3]], (520,120)) # Crop image to ROI and resize
            cv2.imshow('Region of interest', img_roi)

            ocr_result = reader.readtext(img_roi, allowlist=PLATE_CHARS, min_size=60, height_ths=0.1, width_ths=0.25) # Perform OCR
            print(ocr_result)

            print(ocrTextFilter(ocr_result, PLATE_CITIES, threshold=0.7))

            cv2.waitKey(0)

        cv2.destroyAllWindows()

    # reader = easyocr.Reader(['en'], gpu=False)
    # frame = cv2.imread('dataset/images/test/IMG20230221130446.jpg', cv2.IMREAD_COLOR)
    # result = reader.readtext(frame)
    # print(result)

