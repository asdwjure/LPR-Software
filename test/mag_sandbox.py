import cv2
import numpy as np
from matplotlib import pyplot as plt

# image = cv2.imread('/home/jrebernik/Magistrska/LPR-Software/dataset/images_resized/IMG20230220105039.jpg')

# image = cv2.resize(image, (640,480))
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# noise = np.random.normal(0, 25, image.shape)
# image = np.clip(image + noise, 0, 255).astype(np.uint8)

# blurry = cv2.GaussianBlur(image, (7,7), 2)

# image1 = cv2.imread('/home/jrebernik/Pictures/Magistrska_slike/blurry_plate.png')
image = cv2.imread('/home/jrebernik/Pictures/neenakomerno.png', cv2.IMREAD_GRAYSCALE)
image = cv2.GaussianBlur(image, (5,5), 2)

_, bin1 = cv2.threshold(image, 110, 255, cv2.THRESH_BINARY)
bin2 = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 2)

plt.figure()
plt.subplot(131)
plt.axis('off')
plt.title('Originalna slika', y = -0.8, fontsize=10)
plt.imshow(image, cmap='gray')
plt.subplot(132)
plt.axis('off')
plt.title('Upragovanje z eno \n vrednostjo', y = -1.3, fontsize=10)
plt.imshow(bin1, cmap='gray')
plt.subplot(133)
plt.axis('off')
plt.title('Lokalno upragovanje', y = -0.8, fontsize=10)
plt.imshow(bin2, cmap='gray')
plt.savefig('/home/jrebernik/Pictures/Magistrska_slike/primerjava_thr_loc_thr.png', dpi = 300, bbox_inches='tight')
plt.show()


# histogram = cv2.calcHist([img_gamma], [0], None, [256], [0,256], accumulate=False)
# plt.figure()
# plt.subplot(221)
# plt.title('Histogram črnobele slike registrske tablice')
# plt.grid(True)
# plt.xlabel('Vrednost pikslov')
# plt.ylabel('Število pikslov')
# plt.plot(histogram)
# plt.plot(83, histogram[83], 'ro')
# plt.plot(130, histogram[130], 'ro')
# plt.plot(249, histogram[249], 'ro')


# plt.subplot(121)
# plt.axis('off')
# plt.title('Originalna slika', y = -0.7, fontsize=10)
# plt.imshow(image, cmap='gray')
# plt.subplot(122)
# plt.axis('off')
# plt.title("Slika z gamma korekcijo", y = -0.7, fontsize=10)
# plt.imshow(img_gamma, cmap='gray')
# plt.savefig('/home/jrebernik/Pictures/Magistrska_slike/primerjava_gray_gamma.png', dpi = 300, bbox_inches='tight')
# plt.show()