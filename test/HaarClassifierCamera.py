import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = 'C:\Program Files\Tesseract-OCR\\tesseract.exe' # Tesseract path on windows (NOTE: For Linux this must be deleted)

# Set up the USB camera
camera = cv2.VideoCapture(0)

# Load the license plate detection model
plate_cascade = cv2.CascadeClassifier('config/haarcascade_russian_plate_number.xml')

# Loop to capture and process each frame from the camera
while True:
    # Capture a frame from the camera
    ret, frame = camera.read()

    # Convert the frame to grayscale for easier processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect license plates in the frame
    plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Loop through each detected license plate
    for (x, y, w, h) in plates:
        # Crop the license plate region from the frame
        plate_img = frame[y:y+h, x:x+w]

        # Apply some image preprocessing to enhance text recognition
        plate_img = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        plate_img = cv2.GaussianBlur(plate_img, (5, 5), 0)
        plate_img = cv2.threshold(plate_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # Use pytesseract to recognize the text in the license plate image
        plate_text = pytesseract.image_to_string(plate_img, config='--psm 11')

        # Print the recognized text to the console
        print('License plate:', plate_text)

        # Draw a rectangle around the license plate region in the frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the processed frame with license plate regions highlighted
    cv2.imshow('License Plate Recognition', frame)

    # Press 'q' to quit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
camera.release()
cv2.destroyAllWindows()
