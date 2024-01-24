# CUDAatScaleForTheEnterpriseCourseProjectTemplate
A simple CUDA program that reads in an image file, producing a brightened, darkened, and grayscale file as output.

## Project Description

The purpose of this program is to read the image file passed in, then convert it into three separate files.
These are:
- Light: The image is brightened by a percentage the user specifies
- Dark: The image is darkened by a percentage the user specifies
- Grayscale: A grayscale version of the original image

Each of these processes are run in separate CUDA threads, then synced up at the end of the program before exiting with a SUCCESS. If a problem occurs when reading the input file, then the program exits with a FAILURE.

## Code Organization

```bin/```
This folder should hold all binary/executable code that is built automatically or manually. Executable code should have use the .exe extension or programming language-specific extension.

```data/```
This folder should hold all example data in any format. If the original data is rather large or can be brought in via scripts, this can be left blank in the respository, so that it doesn't require major downloads when all that is desired is the code/structure.

```lib/```
Any libraries that are not installed via the Operating System-specific package manager should be placed here, so that it is easier for inclusion/linking.

```src/```
The source code should be placed here in a hierarchical fashion, as appropriate.

```README.md```
This file should hold the description of the project so that anyone cloning or deciding if they want to clone this repository can understand its purpose to help with their decision.

```INSTALL```
This file should hold the human-readable set of instructions for installing the code so that it can be executed. If possible it should be organized around different operating systems, so that it can be done by as many people as possible with different constraints.

```Makefile or CMAkeLists.txt or build.sh```
There should be some rudimentary scripts for building your project's code in an automatic fashion.

```run.sh```
An optional script used to run your executable code, either with or without command-line arguments.