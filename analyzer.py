################################################################
# FILE : analyzer.py
# WRITER : raphael haehnel
# DESCRIPTION: Computes the striations period of a crystal from
#              a microscope image
################################################################

import sys
import os.path
import cv2
import numpy as np
import scipy.fft as sci
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import rotator
from pathlib import Path


''' Kernel for the Sobel filter'''
sobelX = np.array(([3, 0, -3],
                   [4, 0, -4],
                   [3, 0, -3]), dtype="int")


def load_image(path):
    return cv2.imread(path, 0)  # Import an image in grayscale


def rotate_image(img, angle):
    return rotator.apply(img, angle)


def apply_kernel(img, kernel):
    return cv2.filter2D(img, -1, kernel)


def sum_lines_image(img):
    sum_line = img.sum(axis=0)
    return sum_line / np.max(sum_line)  # Function to analyse


def fourier_transform(graph):
    yf = sci.fft(graph, axis=0)  # Analysis of the ideal model
    N = len(graph)
    yf[0:5] = 0  # Neutralize the low frequencies

    xf = np.linspace(0.0, 1.0 / (2.0 * 1), N // 2)   # T = 1
    abs_y = 2.0 / N * np.abs(yf[:N // 2])

    return xf, abs_y


def extract_period(f, y, scale):
    max_amplitude = np.amax(y)
    max_index = np.where(y == max_amplitude)[0][0]
    max_freq = f[max_index]
    real_period = 1.0 / max_freq / scale
    return real_period


def bad_syntax(args):
    for arg in args:
        if bad_syntax_helper(arg):
            return True
    return False


def bad_syntax_helper(arg):
    return arg != "y" and arg != "n"


def char_to_bool(args):
    options = []
    for arg in args:
        if arg == 'y':
            options.append(True)
        else:
            options.append(False)
    return options


def display_graphs(data, options, path, save, scale):
    original_img = data["img"]
    rotated_img = data["img_rotate"]
    filtered_img = data["img_filtered"]
    sum_graph = data["sum_lines"]
    f = data["f"]
    y = data["y"]
    period = data["period"]

    base = os.path.basename(path)
    filename = Path(base)
    filename_wo_ext = str(filename.with_suffix(''))  # Get the path without extension
    out_path = "output/" + filename_wo_ext

    if save and not os.path.exists("output/"):
        os.mkdir("output/")

    if save and not os.path.exists(out_path):
        os.mkdir(out_path)

    if options[0]:  # Displays the original image
        if save:
            cv2.imwrite(out_path + '/1 - original_img' + '.png', original_img)
        else:
            cv2.imshow("{} - opencv".format("original image"), original_img)

    if options[1]:  # Displays the rotated image
        if save:
            cv2.imwrite(out_path + '/2 - rotated_img' + '.png', rotated_img)
        else:
            cv2.imshow("{} - opencv".format("rotated image"), rotated_img)

    if options[2]:  # Displays the filtered image
        if save:
            cv2.imwrite(out_path + '/3 - filtered_img' + '.png', filtered_img)
        else:
            cv2.imshow("{} - opencv".format("filtered image"), filtered_img)

    if options[3]:  # Displays a graph of the intensity of the lines sum
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.set_ylabel("Normalized intensity")
        ax1.set_xlabel("Pixel")
        plt.title("Normalized intensity of the sum of all the lines")
        plt.plot(sum_graph[:len(filtered_img) // 8])

        if save:
            plt.savefig(out_path + '/4 - graph_line' + '.png', dpi=600, format='png')
        else:
            plt.show()

        plt.close()

    if options[4]:  # Displays the lines sum as an image
        sum_img = np.tile(sum_graph, (filtered_img.shape[0], 1))

        if save:
            cv2.imwrite(out_path + '/5 - image_line' + '.png', 255 * sum_img)
        else:
            cv2.imshow("{} - opencv".format("sum lines"), sum_img)

    if options[5]:  # Displays the fourier transform of the lines sum
        fig = plt.figure()
        ax2 = fig.add_subplot(1, 1, 1)

        # Major ticks every 20, minor ticks every 5
        major_ticks = np.arange(0, 8, 0.5)
        minor_ticks = np.arange(0, 8, 0.1)

        ax2.set_xticks(major_ticks)
        ax2.set_xticks(minor_ticks, minor=True)

        # And a corresponding grid
        ax2.grid(which='both')

        # Or if you want different settings for the grids:
        ax2.grid(which='minor', alpha=0.2)
        ax2.grid(which='major', alpha=0.5)

        ax2.set_ylabel("Amplitude")
        ax2.set_xlabel('k vector [1/micron]')

        label_max_freq = 'Lambda = ' + str(period)[:5] + " [micron]"
        blue_patch = mpatches.Patch(color='blue', label=label_max_freq)
        plt.legend(handles=[blue_patch])
        plt.grid(True, which='both', axis='both')
        plt.title("Frequency decomposition of the image")

        f_scaled = f * scale
        plt.plot(f_scaled, y)

        if save:
            plt.savefig(out_path + '/6 - fourier_transform' + '.png', dpi=600, format='png')
            np.savetxt(out_path + "/6 - fourier_transformX.csv", f_scaled, delimiter=',')
            np.savetxt(out_path + "/6 - fourier_transformY.csv", y, delimiter=',')
        else:
            plt.show()

        plt.close()

    cv2.waitKey(0)
    print("Ok !")


def main(path, angle, scale):
    """ The main function. We're calling here all our functions.
    """
    img = load_image(path=path)
    # print("Rotate and filter the image...")
    img_rotate = rotate_image(img=img, angle=angle)
    img_filtered = apply_kernel(img=img_rotate, kernel=sobelX)
    # print("DONE !")
    # print("Sum the lines of the image...")
    sum_lines = sum_lines_image(img=img_filtered)
    # print("DONE !")
    (f, y) = fourier_transform(graph=sum_lines)
    period = extract_period(f, y, scale)

    # print("Period of the striations: ", period, " [um]")
    data = {"img": img,
            "img_rotate": img_rotate,
            "img_filtered": img_filtered,
            "sum_lines": sum_lines,
            "f": f,
            "y": y,
            "period": period}
    return data


if __name__ == "__main__":
    if len(sys.argv) != 8:
        print("ERROR: You need to enter the path and then the 6 options (y/n)")
    elif not os.path.isfile(sys.argv[1]):
        print("ERROR: You need to enter a valid file")
    elif bad_syntax(sys.argv[2:]):
        print("ERROR: Each option has to be defined by yes (y) or no (n)")
    else:
        out_data = main(sys.argv[1], 0, 0)
        display_graphs(out_data, options=char_to_bool(sys.argv[2:]), path=sys.argv[1], save=False)
