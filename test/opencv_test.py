import cv2
import numpy as np
from matplotlib import pyplot as plt
import time


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
        keepWidth = w > 23 and w < 41 # 28-36 plus 5 pixels of margain
        keepHeight = h > 51 and h < 64 # 56-59 plus 5 pixels of margain
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
            
            # construct a mask for the current connected component by
            # finding a pixels in the labels array that have the current
            # connected component ID
            componentMask = (labels == i).astype("uint8") * 255

            cv2.imshow("Connected Component", componentMask)
            cv2.waitKey(0)

    return mask



kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))



img = cv2.imread('dataset/images/train/IMG20230220132839.jpg')
# img = cv2.convertScaleAbs(img, alpha=0.8, beta=-20)
# img = np.power(img/255, 0.8)
# img = (img*255).astype(np.uint8)

# Add noise to image
noise = np.random.normal(0, 25, img.shape)
img = np.clip(img + noise, 0, 255).astype(np.uint8)
# img = cv2.GaussianBlur(img, (15,15), 0)
cv2.imshow('img', img)

plate = img[440:541, 516:926]
plate = cv2.GaussianBlur(plate, (5,5), 0)
plate = cv2.resize(plate, (410,100), interpolation=cv2.INTER_LINEAR) # Ta interpolacija proizvede shitty sliko

gray = np.mean(plate, axis=2).astype(np.uint8) # Raje dej tako v crno belo
gray_mean = np.mean(gray)
cv2.imshow('gray plate', gray)
print('Mean value of gray plate:', gray_mean)

# Apply gamma correction to the image
gamma, img_gamma = gammaCorrection(gray, 110)
plate = cv2.GaussianBlur(plate, (3,3), 0)
cv2.imshow('Gamma corrected plate', img_gamma)
print('Gamma factor =', gamma)

# laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

histogram_gray = cv2.calcHist([gray], [0], None, [256], [0,256], accumulate=False)
histogram_gamma = cv2.calcHist([img_gamma], [0], None, [256], [0,256], accumulate=False)

# _, img_thr = cv2.threshold(img_gamma.copy(), 60, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
img_thr = img_gamma.copy()
img_thr = cv2.adaptiveThreshold(img_thr, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 10)
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
# img_thr = cv2.morphologyEx(img_thr, cv2.MORPH_OPEN, kernel)
# img_thr = cv2.morphologyEx(img_thr, cv2.MORPH_CLOSE, kernel)
cv2.imshow('img thr', img_thr)

plate_chars = plate2chars(img_thr)
plate_chars = cv2.bitwise_not(plate_chars)
cv2.imshow("Characters", plate_chars)



plt.figure(1)
plt.subplot(211)
plt.plot(histogram_gray)
plt.subplot(212)
plt.plot(histogram_gamma)
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
