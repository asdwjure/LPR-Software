import os
import sys
import time
import tensorflow as tf
import cv2
import numpy as np
from matplotlib import pyplot as plt
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
import easyocr

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

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

    category_index = label_map_util.create_category_index_from_labelmap('tensorflow/workspace/annotations/label_map.pbtxt')

    reader = easyocr.Reader(['en'], gpu=False)

    # Load the image using OpenCV
    img = cv2.imread('dataset/images/test/IMG20230220132908.jpg')
    cv2.imshow('Loaded img', img)

    image_np = np.array(img)
    
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)
    
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes']+label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=5,
                min_score_thresh=.8,
                agnostic_mode=False)
    
    cv2.imshow('object detection',  cv2.resize(image_np_with_detections, (800, 600)))
    
    detection_threshold = 0.7
    scores = list(filter(lambda x: x> detection_threshold, detections['detection_scores']))
    boxes = detections['detection_boxes'][:len(scores)]
    classes = detections['detection_classes'][:len(scores)]

    width = image_np_with_detections.shape[1]
    height = image_np_with_detections.shape[0]

    # Apply ROI filtering and OCR
    for idx, box in enumerate(boxes):
        print(box)
        roi = box*[height, width, height, width]
        print(roi)
        region = image_np[int(roi[0]):int(roi[2]),int(roi[1]):int(roi[3])]
        cv2.imshow('Region idx {}'.format(idx), region)
        ocr_result = reader.readtext(region)
        print(ocr_result)
        cv2.waitKey(0)
        # plt.imshow(cv2.cvtColor(region, cv2.COLOR_BGR2RGB))

    for result in ocr_result:
        print(np.sum(np.subtract(result[0][2],result[0][1])))
        print(result[1])
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # reader = easyocr.Reader(['en'], gpu=False)
    # frame = cv2.imread('dataset/images/test/IMG20230221130446.jpg', cv2.IMREAD_COLOR)
    # result = reader.readtext(frame)
    # print(result)

