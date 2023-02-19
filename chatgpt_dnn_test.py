import cv2
import pytesseract
import numpy as np

pytesseract.pytesseract.tesseract_cmd = 'C:\Program Files\Tesseract-OCR\\tesseract.exe' # Tesseract path on windows (NOTE: For Linux this must be deleted)

# Load the pre-trained model
net = cv2.dnn.readNet("lpr.pb")

# Define the classes for the model
classes = ["background", "license plate"]

# Load the image
img = cv2.imread("car.jpg")

# Create a blob from the image
blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), swapRB=True, crop=False)

# Set the input for the model
net.setInput(blob)

# Get the output layer names and run the model
out_layers = net.getUnconnectedOutLayersNames()
outputs = net.forward(out_layers)

# Process the outputs and extract the license plate coordinates
confidences = []
boxes = []
for output in outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5 and class_id == 1:
            center_x, center_y, w, h = (detection[0:4] * np.array([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])).astype('int')
            x = int(center_x - w/2)
            y = int(center_y - h/2)
            confidences.append(float(confidence))
            boxes.append([x, y, int(w), int(h)])

# Draw the license plate on the image
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
for i in indexes:
    i = i[0]
    x, y, w, h = boxes[i]
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Extract the license plate text using Tesseract OCR
plate_img = img[y:y+h, x:x+w]
gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
text = pytesseract.image_to_string(gray, config='--psm 11')

# Print the license plate text
print("License plate: " + text)

# Display the image
cv2.imshow("License Plate Recognition", img)
cv2.waitKey(0)
cv2.destroyAllWindows()