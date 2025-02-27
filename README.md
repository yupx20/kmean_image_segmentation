# Image Segmentation using K-Means Clustering

## Description
This project demonstrates image segmentation using the K-Means clustering algorithm in OpenCV. The input image is converted to the LAB color space for better color differentiation, and segmentation is performed using K-Means clustering.

## Requirements
Make sure you have the following dependencies installed before running the script:

```bash
pip install opencv-python numpy matplotlib
```

## Usage
1. Place an image file (e.g., `test.jpg`) in the same directory as the script.
2. Run the script using Python:

```bash
python image_segmentation.py
```

## Code
Below is the main script used for segmentation:

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Read image file
script_dir = os.path.dirname(os.path.abspath(__file__))
image_filename = "test.jpg"
image_path = os.path.join(script_dir, image_filename)
image = cv2.imread(image_path)

# Convert the color into RGB and LAB
imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
imageLAB = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

# Image pre-processing
pixel_values = imageLAB.reshape((-1, 3))
pixel_values = np.float32(pixel_values)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

# K-Means image segmentation
k = 7  # Adjust the k value for the best segmentation
_, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

centers = np.uint8(centers)
labels = labels.flatten()
segmented_image = centers[labels]
segmented_image = segmented_image.reshape(imageLAB.shape)

# Display the original and segmented image
plt.subplot(211), plt.imshow(imageRGB)
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(212), plt.imshow(segmented_image)
plt.title('Segmented Image'), plt.xticks([]), plt.yticks([])
plt.show()
```

## Results
### Original Image
![Original Image](test.jpg)

### Segmented Image
![Segmented Image](segmented.jpg)

## Adjusting K Value
You can adjust the `k` value in the script to change the segmentation output. Try different values to see how the segmentation varies.