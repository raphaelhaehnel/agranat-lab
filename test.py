# Source
# https://www.pyimagesearch.com/2016/07/25/convolutions-with-opencv-and-python/
# https://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from skimage.io import imread
import cv2
from scipy.signal import convolve2d
import scipy.fft as sci
from skimage.exposure import rescale_intensity
import rotator

# img = cv2.imread('src/132.5, 10.tif', 0)
img = cv2.imread('src/134, 10_2.tif', 0)
scale = 1.0 / 0.0678

# construct average blurring kernels used to smooth an image
tinyBlur = np.ones((3, 3), dtype="float") * (1.0 / (3 * 3))
smallBlur = np.ones((7, 7), dtype="float") * (1.0 / (7 * 7))
largeBlur = np.ones((21, 21), dtype="float") * (1.0 / (21 * 21))

# construct a sharpening filter
sharpen = np.array(([0, -1, 0],
                    [-1, 5, -1],
                    [0, -1, 0]), dtype="int")

# construct the Laplacian kernel used to detect edge-like
# regions of an image
laplacian = np.array(([0, 1, 0],
                      [1, -4, 1],
                      [0, 1, 0]), dtype="int")
# construct the Sobel x-axis kernel
sobelX = np.array(([-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]), dtype="int")

# construct the Sobel y-axis kernel
sobelY = np.array(([-1, -2, -1],
                   [0, 0, 0],
                   [1, 2, 1]), dtype="int")

# construct the kernel bank, a list of kernels we're going
# to apply using both our custom `convole` function and
# OpenCV's `filter2D` function
kernelBank = (
    ("small_blur", smallBlur),
    ("large_blur", largeBlur),
    ("sharpen", sharpen),
    ("laplacian", laplacian),
    ("sobel_x", sobelX),
    ("sobel_y", sobelY)
)

sobelX_v0 = np.array(([6, 1, 0, -1, -6],
                       [8, 3, 0, -3, -8],
                       [10, 3, 0, -3, -10],
                       [8, 3, 0, -3, -8],
                       [6, 1, 0, -1, -6]), dtype="int")

sobelX_v1 = np.array(([10, 6, 0, -6, -10],
                       [10, 6, 0, -6, -10],
                       [10, 6, 0, -6, -10],
                       [10, 6, 0, -6, -10],
                       [10, 6, 0, -6, -10]), dtype="int")

sobelX_v2 = np.array(([10, 8, 6, 0, -6, -8, -10],
                      [10, 8, 6, 0, -6, -8, -10],
                      [10, 8, 6, 0, -6, -8, -10],
                      [10, 8, 6, 0, -6, -8, -10],
                      [10, 8, 6, 0, -6, -8, -10],
                      [10, 8, 6, 0, -6, -8, -10],
                      [10, 8, 6, 0, -6, -8, -10],), dtype="int")

sobelX_v3 = np.array(([-5, -2, 0, 2, 5],
                      [-7, -4, 0, 4, 7],
                      [-7, -6, 0, 6, 7],
                      [-7, -4, 0, 4, 7],
                      [-5, -2, 0, 2, 5]), dtype="int")

sobelX_v4 = np.array(([3, 0, -3],
                      [4, 0, -4],
                      [3, 0, -3]), dtype="int")

myKernelBank = [("sobelX_v0", sobelX_v0),
                ("sobelX_v1", sobelX_v1),
                ("sobelX_v2", sobelX_v2),
                ("sobelX_v3", sobelX_v3)]

# loop over the kernels
# for (kernelName, kernel) in kernelBank:

(kernelName, kernel) = myKernelBank[3]

# apply the kernel to the grayscale image using both
# our custom `convole` function and OpenCV's `filter2D`
# function
print("[INFO] applying {} kernel".format(kernelName))

#######################################
# Apply kernel
# img = cv2.filter2D(img, -1, tinyBlur)
#   opencvOutput2 = cv2.filter2D(img, -1, sobelX_v4)
########################################

#   out_rotated = rotator.apply(opencvOutput2, 359.5)

opencvOutput2 = rotator.apply(img, 359.5)
out_rotated = cv2.filter2D(opencvOutput2, -1, sobelX_v4)

# show the output images
# cv2.imshow("original", img)
# cv2.imshow("{} - opencv".format("sobel_x"), opencvOutput1)

# cv2.imshow("{} - opencv".format("sobelX_v4"), out_rotated)
# cv2.waitKey(0)

# cv2.destroyAllWindows()

(maxY, maxX) = out_rotated.shape
# cv2.imshow("{} - opencv".format("sobelX_v4"), opencvOutput2[0:500, 0:1000])
# cv2.waitKey(0)

# x0 = out_rotated[0, 0:maxX]
# plt.plot(x0)
# plt.ylabel('intensity')
# plt.xlabel('x')
# plt.show()
# x10 = out_rotated[10, 0:maxX]
# x20 = out_rotated[20, 0:maxX]


# f0 = sci.fft(x0)
# plt.plot(np.abs(f0[1:]))
# plt.ylabel('intensity')
# plt.xlabel('f0')
# plt.show()

# f10 = sci.fft(x10)
# plt.plot(np.abs(f10[1:]))
# plt.ylabel('intensity')
# plt.xlabel('f10')
# plt.show()

# f20 = sci.fft(x20)
# plt.plot(np.abs(f20[1:]))
# plt.ylabel('intensity')
# plt.xlabel('f20')
# plt.show()

# Example of fft with sin
#####################################
# N = 600
# T = 1.0/800.0
# x = np.linspace(0.0, N*T, N)
# y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
# yf = sci.fft(y)
# xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
# plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
# plt.show()
#####################################
# cv2.imwrite('./output/out-132.5, 10.png',opencvOutput2)
# cv2.imwrite('./output/outTinyBlur-132.5, 10.png',opencvOutput2)


# Generate image of white and black
width = 2568
height = 1912
PERIOD = 14


# Generate an ideal model for testing results
def get_ideal_model():
    generated_img = np.zeros([height, width, 1], dtype="float32")
    for x in range(maxX):
        for y in range(maxY):
            if (x % PERIOD) < PERIOD / 2:
                generated_img[y, x] = 255
    return generated_img


def ideal_model(save=False):
    generated_img = get_ideal_model()
    # cv2.imshow("{} - opencv".format("my_image"), generated_img)
    # cv2.imwrite('./output/seq.png', generated_img)

    y = generated_img[0, 0:maxX]
    
    # Definition of the scale
    N = maxX
    T = 1

    # Analysis of the ideal model
    yf = sci.fft(y, axis=0)
    xf = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # Major ticks every 20, minor ticks every 5
    major_ticks = np.arange(0, 0.5, 0.05)
    minor_ticks = np.arange(0, 0.5, 0.01)

    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)

    # And a corresponding grid
    ax.grid(which='both')

    # Or if you want different settings for the grids:
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)

    plt.grid(True, which='both', axis='both')
    plt.plot(xf, 2.0 / N * np.abs(yf[0:N // 2]))

    if save:
        plt.savefig('./output/graphFreq-dpi300.png', dpi=300)
    else:
        plt.show()
        cv2.waitKey(0)


# Get the dimensions of the output
(maxY1, maxX1) = out_rotated.shape

# Generate a line of black pixels
sum_line = np.zeros([maxX1], dtype="float32")

# Sum all the lines of the image to one line
for x in range(maxX1):
    sum_pixel = 0.0
    for y in range(maxY1):
        sum_pixel += out_rotated[y, x]
        sum_line[x] = sum_pixel / maxY1

# Function to analyse
sum_line = sum_line / np.max(sum_line)

plt.plot(sum_line[:maxX1 // 8])
plt.show()

sum_img = np.tile(sum_line, (maxY1, 1))


save = False
####################
# Definition of the scale
N = maxX1
T = 1

# Analysis of the ideal model
yf = sci.fft(sum_line, axis=0)

# Neutralize the frequency 0
yf[0] = 0
xf = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

# Major ticks every 20, minor ticks every 5
major_ticks = np.arange(0, 0.5, 0.05)
minor_ticks = np.arange(0, 0.5, 0.01)

ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)

# And a corresponding grid
ax.grid(which='both')

# Or if you want different settings for the grids:
ax.grid(which='minor', alpha=0.2)
ax.grid(which='major', alpha=0.5)

plt.grid(True, which='both', axis='both')
abs_y = 2.0 / N * np.abs(yf[:N // 2])
plt.plot(xf, abs_y)

if save:
    plt.savefig('./output/freq_final.png', dpi=300)
else:
    plt.show()
    cv2.waitKey(0)

cv2.imshow("{} - opencv".format("sobelX_v4"), sum_img)
cv2.waitKey(0)
# cv2.imwrite('./output/sum_img.png', 255 * sum_img)

max_amplitude = np.amax(abs_y)
max_index = np.where(abs_y == max_amplitude)[0][0]
max_freq = xf[max_index]
real_period = 1.0 / max_freq / scale

print(max_amplitude)
print(real_period)
