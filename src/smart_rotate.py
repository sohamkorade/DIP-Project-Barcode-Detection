import cv2
import numpy as np
import matplotlib.pyplot as plt

def smart_rotate(img):
	# # convert to grayscale
	# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# get major axis using hough transform
	edges = cv2.Canny(img, 100, 200)
	lines = cv2.HoughLines(edges, 1, np.pi/180, 200)

	if lines is None:
		return img
	# get angles of lines
	angles = []
	for line in lines:
		for rho, theta in line:
			angles.append(theta)

	# remove very small angles
	# angles = [angle for angle in angles if abs(angle) > 0.1]
	print(angles)
	# get median
	median_angle = np.median(angles)
	print("Median angle: ", median_angle)

	# # plot median line
	# cv2.line(img, (0, 0), (img.shape[1], img.shape[0]), (255, 255, 255), 2)

	# pad image
	img = cv2.copyMakeBorder(img, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=[255, 255, 255])
	# rotate image by angle "median_angle"
	rotated = cv2.warpAffine(img, cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2), median_angle*180/np.pi, 1), (img.shape[1], img.shape[0]))

	# plt.subplot(121),plt.imshow(img, cmap = 'gray')
	# plt.title('Input Image')
	# plt.subplot(122),plt.imshow(rotated, cmap = 'gray')
	# plt.title('Rotated Image')
	# plt.tight_layout()
	# plt.show()

	return rotated