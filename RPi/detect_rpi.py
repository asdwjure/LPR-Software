# Copied from my TFLite repository, and stripped unneeded stuff out

# Import packages
import cv2
import numpy as np
# from tensorflow.lite.python.interpreter import Interpreter
from tflite_runtime.interpreter import Interpreter
import time
import pytesseract
from matplotlib import pyplot as plt

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

    #  if len(ocr) != 8: return None # Plates are of type KPCR-292

    city = ''.join(ocr[0:2]) # first two chars are city
    if city not in PLATE_CITIES:
        return None

    return ''.join(ocr)

def gammaCorrection(img, desired_mean_out):
    mean = np.mean(img)
    gamma = np.log(desired_mean_out/255)/np.log(mean/255)
    img_gamma = np.power(img/255.0, gamma)
    img_gamma = np.uint8(img_gamma*255)
    return gamma, img_gamma

def plate2chars(img, debug=False):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img)

    # initialize an output mask to store all characters parsed from the license plate
    mask = np.zeros(img.shape, dtype="uint8")

    # loop over the number of unique connected component labels
    for i in range(0, num_labels):
        # extract the connected component statistics and centroid for the current label
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        (cX, cY) = centroids[i]
        
        # Only grab components of the appropriate width and height
        keepWidth = w > 7 and w < 47 # 12-36 plus 5 pixels of margain
        keepHeight = h > 40 and h < 67 # 56-62 plus 5 pixels of margain
        # keepArea = area > 500 and area < 1500

        if all((keepWidth, keepHeight)):
            # construct a mask for the current connected component and then take the bitwise OR with the mask
            componentMask = (labels == i).astype("uint8") * 255
            mask = cv2.bitwise_or(mask, componentMask)
        
        if debug:
        # clone our original image (so we can draw on it) and then draw
        # a bounding box surrounding the connected component along with
        # a circle corresponding to the centroid
            output = img.copy()
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.circle(output, (int(cX), int(cY)), 4, (0, 0, 255), -1)

            print('x,y,w,h:', x,y,w,h)
            
            # construct a mask for the current connected component by
            # finding a pixels in the labels array that have the current
            # connected component ID
            componentMask = (labels == i).astype("uint8") * 255
            
            cv2.imshow("Connected Component", componentMask)
            cv2.waitKey(0)

    return mask

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
    cap = cv2.VideoCapture('/home/jrebernik/Magistrska/LPR-Software/test_video.avi')
    # cap = cv2.VideoCapture(0)

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
                img_roi = image[ymin:ymax, xmin:xmax].copy() # Crop image
                img_roi = np.mean(img_roi, axis=2).astype(np.uint8) # Better to convert to grayscale this way as all the colors have the same weight.
                img_roi = cv2.GaussianBlur(img_roi, (5,5), 0) # Filter the noise
                img_roi = cv2.resize(img_roi, (410,100), interpolation=cv2.INTER_LINEAR) # Resize
                cv2.imshow('Gray plate', img_roi)

                gamma, img_roi = gammaCorrection(img_roi, 127) # Correct the gamma
                img_roi = cv2.GaussianBlur(img_roi, (3,3), 0)
                cv2.imshow('Gamma corrected plate', img_roi)
                print('Gamma correction factor = %.2f' % gamma)

                laplacian_var = cv2.Laplacian(img_roi, cv2.CV_64F).var() # Check if image is not blurry
                # TODO: Preveri koliko vpliva na to ce imamo kosmat histogram zaradi gamma korekcije
                # print('laplacian var:', laplacian_var)

                if laplacian_var > 10: # Process only non blurry images
                    # img_roi = cv2.equalizeHist(img_roi)

                    histogram = cv2.calcHist([img_roi], [0], None, [256], [0,256], accumulate=False)

                    if cv2.waitKey(1) & 0xFF ==ord('h'):
                        plt.plot(histogram)
                        plt.show()
                    
                    _, img_roi = cv2.threshold(img_roi, 60, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
                    img_roi = cv2.morphologyEx(img_roi, cv2.MORPH_OPEN, kernel)
                    # img_roi = cv2.morphologyEx(img_roi, cv2.MORPH_CLOSE, kernel)
                    
                    cv2.imshow('Thresholded ROI', img_roi)

                    plate_chars = plate2chars(img_roi, False)
                    plate_chars = cv2.bitwise_not(plate_chars)
                    cv2.imshow("Plate characters", plate_chars)

                    ocr_result = pytesseract.image_to_string(plate_chars, config ='-c load_system_dawg=0 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPRSTUVZYQX0123456789 --psm 11 --oem 1', nice=1)
                    ocr_result = filterTessOutput(ocr_result)
                    
                    if ocr_result != None:
                        # print('Threshold:', hist_max-thr_const)
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

        # All the results have been drawn on the image, now display the image
        cv2.imshow('Image', image)

        # Calculate framerate
        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc= 1/time1

        if cv2.waitKey(10) & 0xFF ==ord('q'):
            cap.release()
            cv2.destroyAllWindows()
        