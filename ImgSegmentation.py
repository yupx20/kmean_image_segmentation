import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Read image file
script_dir = os.path.dirname(os.path.abspath(__file__))
image_filename = "test.jpg"
image_path = os.path.join(script_dir, image_filename)
image = cv2.imread(image_path)

# Convert the color into RGB dan LAB
imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
imageLAB = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

# Image pre-processing
pixel_values = imageLAB.reshape((-1, 3))
pixel_values = np.float32(pixel_values)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

# K-Means image segmentation
k = 7 # Adjust the k value for the best segmentation
_, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

centers = np.uint8(centers)
labels = labels.flatten()
segmented_image = centers[labels]
segmented_image = segmented_image.reshape(imageLAB.shape)

# Saving segmented image to BGR format
segmented_image_bgr = cv2.cvtColor(segmented_image, cv2.COLOR_LAB2BGR)
output_filename = "segmented.jpg"
output_path = os.path.join(script_dir, output_filename)
cv2.imwrite(output_path, segmented_image_bgr)
print(f"Segmented image saved as {output_path}")


# Display the original and segmented image
plt.subplot(211), plt.imshow(imageRGB)
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(212), plt.imshow(segmented_image)
plt.title('Segmented Image'), plt.xticks([]), plt.yticks([])
plt.show()