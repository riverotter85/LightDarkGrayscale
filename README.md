# LightDarkGrayscale
A simple CUDA program that reads in an image file, producing a brightened, darkened, and grayscale file as output.

## Project Description

The purpose of this program is to read in a set of image files passed through the data/ folder. The user is then prompted to specify the percentages for each image to be brightened and darkened, along with a grayscale conversion being done automatically afterward. These are placed into three separate folders as output.

The files are:
- bright_<filename>: The image is brightened by a percentage the user specifies
- dark_<filename>: The image is darkened by a percentage the user specifies
- grayscale_<filename>: A grayscale version of the original image

These processes are done in CUDA kernels in order to help performance, then synced up at the end of the program before exiting. If a problem occurs when reading the input file, the program exits with a FAILURE.

## Building The Code

LightDarkGrayscale requires the latest version of CUDA_Toolkit that's compatible with your GPU. Additionally, you will need to have OpenCV installed.

There are two primary ways that you can install these: either through a baseline Linux distro or Visual Studio for Windows. My personal recommendation is to go with Linux, as the setup procedure is far more streamlined for a developer environment.

NOTE: You MUST use a Linux installation that can speak directly with your computer's GPU. This means that virtualized environments like VmWare and VirtualBox will not work for LightDarkGrayscale T_T

## How To Run

## Light

## Dark

## Grayscale
