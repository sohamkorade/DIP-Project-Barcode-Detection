import cv2
import numpy as np

coords = [0, 1000, 2000, 3000]

coords = np.array(coords)

img = cv2.imread('test1.png')

# draw rectangle on image using various formats

normalize = False

if normalize:
	coords = coords / np.max(coords)

# x1, y1, x2, y2
img = cv2.rectangle(img, (coords[0], coords[1]), (coords[2], coords[3]), (0, 255, 0), 2)

# x1, y1, width, height
img = cv2.rectangle(img, (coords[0], coords[1]), (coords[2] - coords[0], coords[3] - coords[1]), (0, 255, 0), 2)

# x1, y1 (center), width, height
img = cv2.rectangle(img, (coords[0], coords[1]), (coords[2], coords[3]), (0, 255, 0), 2)



import matplotlib.pyplot as plt

plt.imshow(img)
plt.show()
