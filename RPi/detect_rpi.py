# Copied from my TFLite repository, and stripped unneeded stuff out

# Import packages
import cv2
import numpy as np
# from tensorflow.lite.python.interpreter import Interpreter
from tflite_runtime.interpreter import Interpreter
import easyocr
import pytesseract

PATH_TO_MODEL='/home/jrebernik/Magistrska/LPR-Software/RPi/detect.tflite'
PATH_TO_LABELS='labels.txt'
LABELS = ['plate']
MIN_CONF_THRESHOLD=0.8

TEXT_DETECTION_THRESHOLD = 0.5
MIN_AREA_PLATE = 0.05
PLATE_CHARS = 'ABCDEFGHIJKLMNOPRSTUVZYXQ1234567890-' # All possible chars in a plate
PLATE_CITIES = ['KP', 'LJ', 'KR', 'GO', 'PO', 'NM', 'MB', 'SG', 'KK', 'MS', 'CE']

def filterTessOutput(ocr_output):
    ocr = []
    for c in ocr_output:
        if c in PLATE_CHARS:
            ocr.append(c)

    if len(ocr) != 8: return None # Plates are of type KPCR-292

    city = ''.join(ocr[0:2]) # first two chars are city
    if city not in PLATE_CITIES:
        return None

    return ''.join(ocr)

if __name__ == '__main__':

    # Load the Tensorflow Lite model into memory
    interpreter = Interpreter(model_path=PATH_TO_MODEL)
    interpreter.allocate_tensors()

    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    floating_model = (input_details[0]['dtype'] == np.float32)

    input_mean = 127.5
    input_std = 127.5

    # Initialize frame rate calculation
    frame_rate_calc = 1
    freq = cv2.getTickFrequency()
    plate_counter = 0

    # Loop over every image and perform detection
    # cap = cv2.VideoCapture('/home/jrebernik/Magistrska/LPR-Software/dataset/collected_images/VID20230303173709.mp4')
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        # Start timer (for calculating frame rate)
        t1 = cv2.getTickCount()

        # Load image and resize to expected shape [1xHxWx3]
        _, image = cap.read()
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        imH, imW, _ = image.shape 
        image_resized = cv2.resize(image_rgb, (width, height))
        input_data = np.expand_dims(image_resized, axis=0)

        # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std

        # Perform the actual detection by running the model with the image as input
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # Retrieve detection results (detections are sorted, this is why we only take the first one)
        box = interpreter.get_tensor(output_details[1]['index'])[0][0] # Bounding box coordinates of detected objects
        score = interpreter.get_tensor(output_details[0]['index'])[0][0] # Confidence of detected objects

        # If detection box confidence is above minimum threshold
        if score > MIN_CONF_THRESHOLD: # Scores are already sorted

            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1,(box[0] * imH)))
            xmin = int(max(1,(box[1] * imW)))
            ymax = int(min(imH,(box[2] * imH)))
            xmax = int(min(imW,(box[3] * imW)))
            roi_area = (box[2]-box[0])*(box[3]-box[1])

            if roi_area > MIN_AREA_PLATE:
                # print(roi_area)

                img_roi = cv2.resize(image[ymin:ymax, xmin:xmax], (780,195)) # Crop image to ROI and resize
                img_roi = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
                cv2.imshow('Gray plate', img_roi)
                img_roi = cv2.equalizeHist(img_roi)
                img_roi = cv2.GaussianBlur(img_roi, (7,7), 0)
                cv2.imshow('Eq plate', img_roi)

                histogram = cv2.calcHist([img_roi], [0], None, [256], [0,256], accumulate=False)
                hist_max = np.argmax(histogram[50:100], 0)[0] + 50 # Najdi peak med pixli ki imajo vrednost med 50 in 100


                for thr_const in [5, 15, 25]:
                    _, img_roi = cv2.threshold(img_roi, hist_max-thr_const, 255, cv2.THRESH_BINARY)
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                    img_roi = cv2.morphologyEx(img_roi, cv2.MORPH_ERODE, kernel=kernel)
                    cv2.imshow('Region of interest', img_roi)

                    ocr_result = pytesseract.image_to_string(img_roi)
                    ocr_result = filterTessOutput(ocr_result)
                    
                    if ocr_result != None:
                        print('Threshold:', hist_max-thr_const)
                        print(plate_counter, ocr_result)
                        plate_counter += 1

                cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (0,255,0), 2)

                # Draw label
                object_name = 'plate' # Look up object name from "labels" array using class index
                label = '%s: %d%%' % (object_name, int(score*100)) # Example: 'person: 72%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                cv2.rectangle(image, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                cv2.putText(image, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

        # Draw framerate in corner of frame
        cv2.putText(image,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

        # Calculate framerate
        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc= 1/time1

        # All the results have been drawn on the image, now display the image
        cv2.imshow('Image', image)

        if cv2.waitKey(10) & 0xFF ==ord('q'):
            cap.release()
            cv2.destroyAllWindows()
        