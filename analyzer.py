################################################################
# FILE : analyzer.py
# WRITER : Raphael Haehnel
# DESCRIPTION: Computes the striations period of a crystal from
#              a microscope image
################################################################

import sys
import os.path
from cv2 import filter2D, imshow, waitKey, imdecode, imencode, IMREAD_GRAYSCALE
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
    """
    Decode the path and load the image. Support unicode characters
    To read an image in grayscale without decoding, use "return imread(path, 0)"
    :param path: the path of the image
    :return: a numpy array representing an image in grayscale
    """
    print("path = " + path)
    stream = open(path, "rb")
    bytes_arr = bytearray(stream.read())
    np_arr = np.asarray(bytes_arr, dtype=np.uint8)
    return imdecode(np_arr, IMREAD_GRAYSCALE)


def rotate_image(img, angle):
    """
    Uses the function from the script "rotator.py"
    :param img: a numpy array representing an image in grayscale
    :param angle: the angle to rotate the image
    :return: the image rotated
    """
    return rotator.apply(img, -angle)


def apply_kernel(img, kernel):
    """
    Uses the function from the script "rotator.py"
    :param img: a numpy array representing an image in grayscale
    :param kernel: a numpy array representing a kernel
    :return: the image filtered
    """
    return filter2D(img, -1, kernel)


def sum_lines_image(img):
    """
    Sums each line with the others of the image
    :param img: a numpy array representing an image in grayscale
    :return: an array representing the sum of the lines
    """
    sum_line = img.sum(axis=0)
    return sum_line / np.max(sum_line)  # Function to analyse


def fourier_transform(graph):
    """
    Computes the fourier transform of a function
    :param graph: An array of values
    :return: A tuples of two arrays: f and y
    """
    yf = sci.fft(graph, axis=0)  # Analysis of the ideal model
    N = len(graph)
    yf[0:5] = 0  # Neutralize the low frequencies

    xf = np.linspace(0.0, 1.0 / (2.0 * 1), N // 2)   # T = 1
    abs_y = 2.0 / N * np.abs(yf[:N // 2])

    return xf, abs_y


def extract_period(f, y, scale):
    """
    Find the maximal value of a graph
    :param f: the abscissa of the graph
    :param y: the ordinate of the graph
    :param scale: the relation between x and the real value
    :return: the value of x that corresponds to the maximum
    """
    max_amplitude = np.amax(y)
    max_index = np.where(y == max_amplitude)[0][0]
    max_freq = f[max_index]
    real_period = 1.0 / max_freq / scale
    return real_period


def bad_syntax(args):
    """
    Checks the syntax of the arguments
    :param args: an array of strings
    :return: A boolean
    """
    for arg in args:
        if bad_syntax_helper(arg):
            return True
    return False


def bad_syntax_helper(arg):
    """
    Checks the syntax of the arguments
    :param args: an array of char
    :return: A boolean
    """
    return arg != "y" and arg != "n"


def char_to_bool(args):
    """
    Convert an array of chars (y/n) to booleans
    :param args: an array of char
    :return: an array of booleans
    """
    options = []
    for arg in args:
        if arg == 'y':
            options.append(True)
        else:
            options.append(False)
    return options


def display_graphs(data, options, path, out, save, scale):
    """
    Displays or saves the data according to the options
    :param data: a struct containing the data
    :param options: an array of booleans
    :param path: the path of the original image
    :param out: the path of the output images
    :param save: a boolean
    :param scale: the relation between a pixel and the real distance
    :return: None
    """
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
    out_path = out + '/' + filename_wo_ext

    if save and not os.path.exists(out):
        os.mkdir(out)

    if save and not os.path.exists(out_path):
        os.mkdir(out_path)
    if options[0]:  # Displays the original image
        if save:
            is_success, im_buf_arr = imencode('.png', original_img)  # Encode the path to support unicode characters
            im_buf_arr.tofile(out_path + '/1 - original_img' + '.png')
            # imwrite(out_path + '/1 - original_img' + '.png', original_img)
        else:
            imshow("{} - opencv".format("original image"), original_img)

    if options[1]:  # Displays the rotated image
        if save:
            is_success, im_buf_arr = imencode('.png', rotated_img)
            im_buf_arr.tofile(out_path + '/2 - rotated_img' + '.png')
        else:
            imshow("{} - opencv".format("rotated image"), rotated_img)

    if options[2]:  # Displays the filtered image
        if save:
            is_success, im_buf_arr = imencode('.png', filtered_img)
            im_buf_arr.tofile(out_path + '/3 - filtered_img' + '.png')
        else:
            imshow("{} - opencv".format("filtered image"), filtered_img)

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
            is_success, im_buf_arr = imencode('.png', 255 * sum_img)
            im_buf_arr.tofile(out_path + '/5 - image_line' + '.png')
        else:
            imshow("{} - opencv".format("sum lines"), sum_img)

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
        ax2.set_xlabel('1/Lambda [1/micron]')

        label_max_freq = 'Lambda = ' + str(period)[:5] + " [micron]"
        blue_patch = mpatches.Patch(color='blue', label=label_max_freq)
        plt.legend(handles=[blue_patch])
        plt.grid(True, which='both', axis='both')
        plt.title("Frequency decomposition of the image")

        f_scaled = f * scale
        plt.plot(f_scaled, y)

        if save:
            plt.savefig(out_path + '/6 - fourier_transform' + '.png', dpi=600, format='png')
            np.savetxt(out_path + "/6 - fourier_transform.csv", (f_scaled, y), delimiter=',', fmt='%.4f')
        else:
            plt.show()

        plt.close()

    waitKey(0)


def main(path, angle, scale):
    """ The main function. We're calling here all our functions.
    """
    img = load_image(path=path)
    img_rotate = rotate_image(img=img, angle=angle)
    img_filtered = apply_kernel(img=img_rotate, kernel=sobelX)
    sum_lines = sum_lines_image(img=img_filtered)
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
    """ An test function
    """
    if len(sys.argv) != 8:
        print("ERROR: You need to enter the path and then the 6 options (y/n)")
    elif not os.path.isfile(sys.argv[1]):
        print("ERROR: You need to enter a valid file")
    elif bad_syntax(sys.argv[2:]):
        print("ERROR: Each option has to be defined by yes (y) or no (n)")
    else:
        out_data = main(sys.argv[1], 0, 0)
        display_graphs(out_data, options=char_to_bool(sys.argv[2:]), path=sys.argv[1], out_path="../output/", save=False)
