# Import required libraries
import tensorflow as tf
from PIL import Image

# Read a PIL image  
img = Image.open('/home/jrebernik/Magistrska/LPR-Software/dataset/images_resized/IMG20230220105031.jpg')
print(img)
# Convert the PIL image to Tensor
img_to_tensor = tf.convert_to_tensor(img)
# print the converted Torch tensor
print(img_to_tensor)
print(len(img_to_tensor))
print("dtype of tensor:",img_to_tensor.dtype)