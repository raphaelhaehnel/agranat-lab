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

