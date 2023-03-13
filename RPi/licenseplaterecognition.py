import cv2
import numpy as np
# from tensorflow.lite.python.interpreter import Interpreter
from tflite_runtime.interpreter import Interpreter
import time
import pytesseract
from matplotlib import pyplot as plt
import sys
sys.path.append('/home/jrebernik/Magistrska/LPR-Software/webapp')
from webapp import LPR_Webapp


class LicensePlateRecognition:
    
    def __init__(self, stream_queue, params=None, debug_level=0):

        self.stream_queue = stream_queue

        # Construct parameters dictionary
        self.params = {
            'model_path' : params['model_path'],
            'capture_source' : params['capture_source'],
            'min_conf_threshold' : params['min_conf_threshold'],
            'min_area_plate' : params['min_area_plate'],
            'min_laplacian_var' : params['min_laplacian_var'],
            'plate_chars' : params['plate_chars'],
            'plate_cities' : params['plate_cities']
        }
        self.debug_level = debug_level

        # Load the Tensorflow Lite model into memory
        self.interpreter = Interpreter(model_path=params['model_path'])
        self.interpreter.allocate_tensors()

        # Get model details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.height = self.input_details[0]['shape'][1]
        self.width = self.input_details[0]['shape'][2]

        self.floating_model = (self.input_details[0]['dtype'] == np.float32)

        self.input_mean = 127.5
        self.input_std = 127.5

        # Initialize frame rate calculation
        self.frame_rate_calc = 1
        self.freq = cv2.getTickFrequency()
        self.plate_counter = 0

        # Loop over every image and perform detection
        if params['capture_source'] == 0:
            self.cap = cv2.VideoCapture(0)
        else:
            self.cap = cv2.VideoCapture(params['capture_source'])

    def __filterTessOutput(self, raw_ocr):
        """Filter output from PyTesseract.
        Check if characters in ocr_output are in PLATE_CHARS. Also check if
        first two characters are a valid city.
        
        @param
            ocr_output: raw output from PyTesseract
            
        @return
            Filtered license plate (string)"""
        
        if self.debug_level >= 2:
            print("Raw OCR output: '%s'" % raw_ocr) # Print raw OCR output as debug information

        filtered_ocr = []
        for c in raw_ocr:
            if c in self.params['plate_chars']:
                filtered_ocr.append(c)

        city = ''.join(filtered_ocr[0:2]) # first two chars are city
        if city not in self.params['plate_cities']:
            return None

        return ''.join(filtered_ocr)

    def __gammaCorrection(self, img, desired_mean_out):
        """Gamma correction.
        Calculate gamma factor based on desired_mean_out value and apply gamma correction.
        
        @param:
            img: Source image.
            desired_mean_out: Desired mean value of the output image.
            
        @return
            Gamma corrected image."""
        
        mean = np.mean(img)
        gamma = np.log(desired_mean_out/255)/np.log(mean/255)
        img_gamma = np.power(img/255.0, gamma)
        img_gamma = np.uint8(img_gamma*255)

        if self.debug_level: cv2.imshow('Gamma corrected plate', img_gamma)
        if self.debug_level >= 2: print('Gamma correction factor = %.2f' % gamma)

        return gamma, img_gamma

    def __plate2chars(self, img):
        """Extract and return a mask where the characters are in a plate.
        Mask elements need to be of a cerating width and height in order to be considered valid.

        @param
            img: Binary image of white characters on a black background.
            debug: Optionally provide debug information.
            
        @return
            Mask of white characters on black background."""
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img)

        # initialize an output mask to store all characters parsed from the license plate
        mask = np.zeros(img.shape, dtype="uint8")

        output = img.copy() # For debugging
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)

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
            keepWidth = w > 7 and w < 48 # 12-36 plus 5 pixels of margain
            keepHeight = h > 40 and h < 68 # 56-62 plus 5 pixels of margain
            keepX = x > 40 # Eliminate SLO area
            # keepArea = area > 500 and area < 1500

            if all((keepWidth, keepHeight, keepX)):
                # construct a mask for the current connected component and then take the bitwise OR with the mask
                componentMask = (labels == i).astype("uint8") * 255
                mask = cv2.bitwise_or(mask, componentMask)
            
                if self.debug_level:
                # clone our original image (so we can draw on it) and then draw
                # a bounding box surrounding the connected component along with
                # a circle corresponding to the centroid
                    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.circle(output, (int(cX), int(cY)), 3, (0, 0, 255), -1)

                    if self.debug_level >= 3:
                        print('Mask component {}: (x,y,w,h) = {}'.format(i, (x,y,w,h)))


        mask = cv2.bitwise_not(mask) # We need black characters on white surface

        if self.debug_level:
            cv2.imshow("Boxes of connected components", output)
            cv2.imshow("Plate characters mask", mask)

        return mask

    def process_image(self):
        """Needs to be running in a separate process because this is an infinate loop."""

        while True:
            # Start timer (for calculating frame rate)
            t1 = cv2.getTickCount()

            # Load image and resize to expected shape [1xHxWx3]
            _, image = self.cap.read()
            # noise = np.random.normal(0, 25, image.shape)
            # image = np.clip(image + noise, 0, 255).astype(np.uint8)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            imH, imW, _ = image.shape 
            image_resized = cv2.resize(image_rgb, (self.width, self.height))
            input_data = np.expand_dims(image_resized, axis=0)

            # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
            if self.floating_model:
                input_data = (np.float32(input_data) - self.input_mean) / self.input_std

            # Perform the actual detection by running the model with the image as input
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()

            # Retrieve detection results (detections are sorted, this is why we only take the first one)
            box = self.interpreter.get_tensor(self.output_details[1]['index'])[0][0] # Bounding box coordinates of detected objects
            score = self.interpreter.get_tensor(self.output_details[0]['index'])[0][0] # Confidence of detected objects

            # If detection box confidence is above minimum threshold
            if score > self.params['min_conf_threshold']: # Scores are already sorted

                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                ymin = int(max(1,(box[0] * imH)))
                xmin = int(max(1,(box[1] * imW)))
                ymax = int(min(imH,(box[2] * imH)))
                xmax = int(min(imW,(box[3] * imW)))
                roi_area = (box[2]-box[0])*(box[3]-box[1])
                if self.debug_level >= 2: print('ROI area = %.4f' % roi_area)

                if roi_area > self.params['min_area_plate']:
                    img_roi = image[ymin:ymax, xmin:xmax].copy() # Crop image
                    img_roi = np.mean(img_roi, axis=2).astype(np.uint8) # Better to convert to grayscale this way as all the colors have the same weight.
                    img_roi = cv2.GaussianBlur(img_roi, (5,5), 0) # Filter the noise
                    img_roi = cv2.resize(img_roi, (410,100), interpolation=cv2.INTER_LINEAR) # Resize. TODO: Possible tweak: make the ROI image bigger so we have more information. This way we would avoid thresholding issues between two chars.
    
                    if self.debug_level: cv2.imshow('Gray plate', img_roi)

                    laplacian_var = cv2.Laplacian(img_roi, cv2.CV_64F).var() # Check if image is not blurry
                    if self.debug_level >= 2: print('Laplacian var = %.2f' % laplacian_var)

                    if laplacian_var > self.params['min_laplacian_var']: # Process only non blurry images

                        _, img_roi = self.__gammaCorrection(img_roi, 127) # Correct the gamma
                        # img_roi = cv2.GaussianBlur(img_roi, (3,3), 0) # TODO: Play around with this if it is neeed. I would assume that it is.

                        histogram = cv2.calcHist([img_roi], [0], None, [256], [0,256], accumulate=False)
                        if cv2.waitKey(1) & 0xFF ==ord('h'): # Press H to display histogram
                            plt.plot(histogram)
                            plt.show()
                        
                        img_roi = cv2.adaptiveThreshold(img_roi, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 2)
                        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
                        img_roi = cv2.morphologyEx(img_roi, cv2.MORPH_OPEN, kernel)
                        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
                        img_roi = cv2.morphologyEx(img_roi, cv2.MORPH_CLOSE, kernel)
                        
                        if self.debug_level: cv2.imshow('Thresholded ROI', img_roi)

                        plate_chars = self.__plate2chars(img_roi)

                        ocr_result = pytesseract.image_to_string(plate_chars, config ='-c load_system_dawg=0 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPRSTUVZYQX0123456789 --psm 6 --oem 1', nice=1)
                        ocr_result = self.__filterTessOutput(ocr_result)
                        
                        if ocr_result != None:
                            print("Detected license plate #{}: {}".format(self.plate_counter, ocr_result))
                            self.plate_counter += 1

                cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (0,255,0), 2)
                # Draw label
                object_name = 'plate' # Look up object name from "labels" array using class index
                label = '%s: %d%%' % (object_name, int(score*100)) # Example: 'plate: 72%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                cv2.rectangle(image, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                cv2.putText(image, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

            # Draw framerate in corner of frame
            cv2.putText(image,'FPS: {0:.2f}'.format(self.frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
            
            if self.debug_level:
                # All the results have been drawn on the image, now display the image
                cv2.imshow('Image', image)

            # Calculate framerate
            t2 = cv2.getTickCount()
            time1 = (t2-t1)/self.freq
            self.frame_rate_calc= 1/time1

            LPR_Webapp.put_frame(image)
            # self.stream_queue.put(image, False)

            if cv2.waitKey(10) & 0xFF ==ord('q'):
                self.cap.release()
                cv2.destroyAllWindows()
