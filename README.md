# Project description

This program is a tool to analyze microscope images.
  
It is able to take an image of a crystal than contains a bragg grating,
and computes the period of this grating. It uses the OpenCV library to proccess the image.

## Usage

You can see below the main window.

![mainWindow](https://user-images.githubusercontent.com/69756617/205871339-29e389ab-eab8-4456-ab03-40e7dea158ae.png)

### Define parameters

Before you are starting your work, you need to define some parameters, by pressing File > Define parameters.
Scale: Define the length (in micrometers) of a single pixel
Rotation angle: Define the angle for which we must rotate the image to make the lines of the grating parallel to the image borders

![parameters](https://user-images.githubusercontent.com/69756617/205871420-f4fc7932-7cf5-41ee-9618-cb70e4e4c431.png)

The rotation angle can be defined by yourself.
It have to be precise, so the computation of the grating period can be reliable. However, you can find the rotation angle automatically by pressing the button "Angle finder". But you still need to define a range angle and a resolution so the program can find the optimal angle.

![automatic rotation](https://user-images.githubusercontent.com/69756617/205871732-e7f6edd7-01aa-4294-96fa-efe40f69515c.png)

## Input

The input of the algorithm is a photo taken by a microscope. On the image, you can see many lines in different directions and imperfections. The lines that are interesting use are the parallel lines to the y-axis. This is the Bragg grating we are looking to characterize.

## Output

### Rotation

The first step is to rotate the image so the grating is **perfectly** parallel to the y-axis. This is a requirement for a correct computation.

![2 - rotated_img](https://user-images.githubusercontent.com/69756617/205872947-097589af-c38e-4e1b-ac7f-23ebf3f27679.png)

As a second step, we apply a linear filter, the SobelX kernel, to find horizontal edges.

![3 - filtered_img](https://user-images.githubusercontent.com/69756617/205872956-991da445-9188-4b76-9997-42d9fcb135a2.png)

Next, we are summing all the pixels along the y-axis. In result, we got 1D array of values, where each value is the "mean" of each column.

![4 - graph_line](https://user-images.githubusercontent.com/69756617/205872966-663155d4-862b-43bc-b5a6-08748c8f547a.png)

To help us visualize the bragg grating as an image, we can expand the single-line array and duplicate the line throughout the y-axis. The image we get is the same as the previous graph, but in 2D.

![5 - image_line](https://user-images.githubusercontent.com/69756617/205872980-2cf47c61-fb89-439d-ba4e-84bb4cf55647.png)

All that is left is to compute the fourier transform of the graph, and extract from it the maximul value, that correspond to the bragg gratings. If the bragg grating was perfect and without noise, we would have get a delta function.

![6 - fourier_transform](https://user-images.githubusercontent.com/69756617/205872991-b56252e0-9ff0-41ab-a0be-53e5c8d6d78d.png)
