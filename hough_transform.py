import numpy as np

from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny
from skimage.draw import line
from skimage import data

import matplotlib.pyplot as plt
from matplotlib import cm
import analyzer


''' Kernel for the Sobel filter'''
blur = np.array(([1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1]), dtype="int")

''' Kernel for the Sobel filter'''
other = np.array(([2, 1, 1, 1, -2],
                  [3, 1, 1, 1, -3],
                  [4, 1, 1, 1, -4],
                  [3, 1, 1, 1, -3],
                  [2, 1, 1, 1, -2]), dtype="int")


# Constructing test image
"""
image = np.zeros((200, 200))
idx = np.arange(25, 175)
image[idx, idx] = 255
image[line(45, 25, 25, 175)] = 255
image[line(25, 135, 175, 155)] = 255
"""

image = analyzer.load_image("H:/Raphael - Docs/Archives/Universités/Hebrew University/Matériel de cours/2021 - mehkar_student/image_processing/src/132.5, 10 resized.png")
image = analyzer.apply_kernel(image, analyzer.sobelX)
image = analyzer.apply_kernel(image, other)

# Classic straight-line Hough transform
# Set a precision of 0.5 degree.
bound = np.pi / 20
N = 30
precision = 2 * bound / N * 180 / np.pi
print("precision = ", precision)
tested_angles = np.linspace(-bound, bound, N, endpoint=False)
h, theta, d = hough_line(image, theta=tested_angles)
print("h = ", h)

# Generating figure 1
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
ax = axes.ravel()

ax[0].imshow(image, cmap=cm.gray)
ax[0].set_title('Input image')
ax[0].set_axis_off()

angle_step = precision * np.diff(theta).mean()
d_step = precision * np.diff(d).mean()
bounds = [np.rad2deg(theta[0] - angle_step),
          np.rad2deg(theta[-1] + angle_step),
          d[-1] + d_step, d[0] - d_step]
ax[1].imshow(np.log(1 + h), extent=bounds, cmap=cm.gray, aspect=1 / 1.5)
ax[1].set_title('Hough transform')
ax[1].set_xlabel('Angles (degrees)')
ax[1].set_ylabel('Distance (pixels)')
ax[1].axis('image')

# ax[2].imshow(image, cmap=cm.gray)
# ax[2].set_ylim((image.shape[0], 0))
# ax[2].set_axis_off()
# ax[2].set_title('Detected lines')

# for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
#     (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
#     ax[2].axline((x0, y0), slope=np.tan(angle + np.pi/2))

plt.tight_layout()
plt.show()