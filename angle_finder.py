################################################################
# FILE : angle_finder.py
# WRITER : Azriel Gold
# DESCRIPTION: Find the appropriate ange to rotate the image
################################################################

import numpy as np
from scipy.ndimage import rotate
from scipy.signal import fftconvolve
from skimage.draw import line_aa
import matplotlib.pyplot as plt
import analyzer


class AutoRotator:
    """Class for registering images using blood vessel segmentation"""
    def __init__(self, path, rotate_parameters):
        """
        Class constructor
        :param bl_image: bl image object
        :param fu_image: fu image object
        """
        self.path = None
        self.original = None
        self.model = None
        self.original = analyzer.load_image(path)
        self.model = np.zeros(self.original.shape, dtype=np.uint8)
        rr, cc, val = line_aa(0, int(self.model.shape[1] / 2), self.model.shape[0] - 1, int(self.model.shape[1] / 2))
        self.model[rr, cc] = val * 255
        self.rotate_parameters = rotate_parameters
        # self.graph = np.zeros((2, 100))

    # def __add_caption_area(self, grayscale_img, caption_footer_size=100):
    #     """
    #     Add black area where caption goes
    #     :param grayscale_img: single channel grayscale image
    #     :return: image with black space as footer
    #     """
    #     full_img = np.zeros((grayscale_img.shape[0] + caption_footer_size, grayscale_img.shape[1]))
    #     full_img[:grayscale_img.shape[0]] = grayscale_img
    #     return full_img

    def calc_optimal_transformation(self, progress_callback):
        """
        Calculate optimal rigid transformation
        :return: fu image rotation angle and translation vector
        """
        highest_max_value = (0, 0)
        max_val_coordinates = (0, 0)
        # i = 0
        # cycle through all relevant angles and find best one
        min_angle = self.rotate_parameters[0]
        max_angle = self.rotate_parameters[1]
        resolution = self.rotate_parameters[2]
        angles = np.round(np.arange(min_angle, max_angle, resolution), 3)

        i = 0
        max = angles.size - 1

        for angle in angles:

            print(f'\rSearching for optimal angle. Current best is {highest_max_value[1]}°. Checking {angle}°...', end='')

            if progress_callback:
                progress_callback.emit([int(i / max * 100), highest_max_value[1], angle])
                i += 1

            rotated_fu_seg = rotate(self.model, angle)
            cross_correlation = fftconvolve(self.original, np.flip(np.flip(rotated_fu_seg, axis=1), axis=0), mode='same')
            max_white_value = np.max(cross_correlation)
            # self.graph[0][i] = angle
            # self.graph[1][i] = max_white_value
            # i = i + 1
            # check if current angle is best

            if max_white_value > highest_max_value[0]:
                highest_max_value = (max_white_value, angle)
                max_val_coordinates = cross_correlation.argmax()
                max_val_coordinates = (max_val_coordinates % cross_correlation.shape[1], max_val_coordinates // cross_correlation.shape[1])

        print(f'\rSearching for optimal angle...Done ({highest_max_value[1]}° is best angle)\n', end='')

        if progress_callback:
            progress_callback.emit([int(i / max * 100), highest_max_value[1], None])
            i += 1

        # get image center and calculate translation vector based on that and based on max value of cross correlation of chosen angle
        img_center = np.array(self.original.shape) / 2
        translation_vector = img_center[1] - max_val_coordinates[0], img_center[0] - max_val_coordinates[1]

        # plot image center and max cross correlation point - uncomment for visualization
        # import matplotlib.pyplot as plt
        # plt.imshow(self.__perform_cross_correlation(rotate(self.model, highest_max_value[1])))
        # plt.scatter(img_center[1], img_center[0])
        # plt.scatter(max_val_coordinates[0], max_val_coordinates[1])
        # plt.axis('off')
        # plt.show()

        # fig = plt.figure()
        # ax1 = fig.add_subplot(1, 1, 1)
        # ax1.set_ylabel("Correlation")
        # ax1.set_xlabel("Angle")
        # plt.title("Correlation of the images")
        # plt.plot(self.graph[0], self.graph[1])
        # plt.show()

        # return rotation angle and translation vector

        return highest_max_value[1], translation_vector

    def run_registration_engine(self):
        """
        Calculate FU image transformation matrix using __calc_optimal_transformation(), and
        builds transformation matrix
        :return: calculated transformation matrix
        """
        # get optimal angle and translation vector
        (angle, translation) = self.calc_optimal_transformation

        # calculate rotation
        theta = np.radians(angle)
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)

        # build transformation matrix
        transformation_matrix = np.eye(3)
        transformation_matrix[0: 2] = [[cos_theta, -sin_theta, translation[0]],
                                       [sin_theta, cos_theta,  translation[1]]]
        return transformation_matrix


# if __name__ == "__main__":
#     my_obj = AutoRotator()
#     my_obj.run_registration_engine()

