import cv2
import numpy as np


kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))



img = cv2.imread('dataset/images/test/IMG20230220132908.jpg')
plate = img[399:523, 360:630]
test = plate[50:60, 0:200]
print(np.min(test))
cv2.imshow('test', test)
gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7,7), 0)

eq = cv2.equalizeHist(gray)

cv2.imshow('Original', gray)
cv2.imshow('Eq', eq)

min_value = np.min(eq)
mean_value = np.mean(eq)
mean_gray = np.mean(gray)
print(min_value, mean_value, mean_gray)
_, binary = cv2.threshold(eq, mean_value-50, 255, cv2.THRESH_BINARY)
cv2.imshow('thr img', binary)
binary = cv2.erode(binary, kernel=kernel)
# binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
cv2.imshow('eroded', binary)

# binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 15)
# _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# cv2.imshow('thr img', binary)


# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow('gray after thr', gray)
# inverted = cv2.bitwise_not(binary)
# cv2.imshow('mask', inverted)
# masked = cv2.bitwise_and(img, img, mask=inverted)
# cv2.imshow('Masked img', masked)

# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# gray = cv2.GaussianBlur(gray, (7,7), 0)

# gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
# cv2.imshow('Gray threshold', gray)

# img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

# plate = gray[399:523, 360:630]
# cv2.imshow('Plate', plate)

cv2.waitKey(0)
cv2.destroyAllWindows()
