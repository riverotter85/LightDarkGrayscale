# LightDarkGrayscale
A simple CUDA program that reads in an image file, producing a brightened, darkened, and grayscale file as output.

## Description

The purpose of this program is to read in a set of image files passed through the ```data/``` folder. The user is then prompted to specify the percentages for each image to be brightened and darkened, along with a grayscale conversion being done automatically afterward. These are placed into three separate folders as output.
- ```bright_<filename>```: The image is brightened by a percentage the user specifies
- ```dark_<filename>```: The image is darkened by a percentage the user specifies
- ```grayscale_<filename>```: A grayscale version of the original image

These processes are done in CUDA kernels in order to help performance, then synced up at the end of the program before exiting. If a problem occurs when reading the input file, the program exits with a FAILURE.

## Building The Code

LightDarkGrayscale requires the latest version of CUDA_Toolkit that's compatible with your GPU. Additionally, you will need to have OpenCV installed.

There are two primary ways you can install these: either through a baseline Linux distro or Visual Studio for Windows. My personal recommendation is to go with Linux, as the setup procedure is far more streamlined for a developer environment.

Install CUDA by following this link: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html

Install OpenCV using one of the following two methods:
- Install ```libopencv-dev``` using your system's package manager (Linux only)
- Follow the instructions on OpenCV's website
    - Windows: https://docs.opencv.org/4.x/d3/d52/tutorial_windows_install.html
    - Linux: https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html

NOTE: You MUST use a Linux installation that can interact with your computer's GPU. This means virtualized environments like VmWare and VirtualBox will NOT work for LightDarkGrayscale T_T

## How To Run

Once LightDarkGrayscale is built, simply run the executable in the LightDarkGrayscale folder, using the ```./bin/LightDarkGrayscale.exe``` command. You can also execute the run.sh file, using ```./run.sh```. This will run the program without any further interaction required from the user, and the output will be logged into output.txt.

Once finished, you will be able to see the brightened, darkened, and grayscaled files by checking the contents of ```data/output/```. The files themselves will be separated into that of "bright", "dark", and "grayscale", respectively.

## Arguments

LightDarkGrayscale has two arguments which you can use to help speed up the runtime process.
- ```-b <percentage>```: Pass in the brighten percentage as an argument
- ```-d <percentage>```: Pass in the darken percentage as an argument