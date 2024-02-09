// File: LightDarkGrayscale.cu
// Author: Logan Davis
// Created: 1/24/2024
// Last Modified: 2/09/2024
//
// NOTE: This project was originally cloned from the template "CUDAatScaleForTheEnterpriseCourseProjectTemplate", which was provided by Chancellor Pascale.
//       The original code can be found at the link: https://github.com/PascaleCourseraCourses/CUDAatScaleForTheEnterpriseCourseProjectTemplate
//
// *** LightDarkGrayscale ***
//
// A simple CUDA program that reads in an image file, producing a brightened, darkened, and grayscale file as output.

#include <iostream>
#include <string>
#include <filesystem>
#include <list>
#include <exception>
#include <vector>
#include <climits>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include "CudaImage.hpp"

using namespace std;

// Device function to calculate brightened color pixel
// Arguments:
// - color (uchar): Original pixel color
// - percentage (int): Percentage by which the pixel is brightened
// Returns: uchar
__device__ uchar brightenColor(uchar color, int percentage) {
    float modifier = 1 + percentage * 0.01;
    uchar temp = color;

    uchar newColor = color * modifier;
    if (newColor < temp) newColor = UCHAR_MAX; // Byte data wrapped around; set as MAX value

    return newColor;
}

// Device function to calculate darkened color pixel
// Arguments:
// - color (uchar): Original pixel color
// - percentage (int): Percentage by which the pixel is darkened
// Returns: uchar
__device__ uchar darkenColor(uchar color, int percentage) {
    float modifier = 1 - percentage * 0.01;
    return color * modifier;
}

// Kernel function to calculate and set the filtered pixels for brightened, darkened, and grayscaled images
// Arguments:
// - d_r                (uchar*): Device array for color r (red) pixels
// - d_g                (uchar*): Device array for color g (green) pixels
// - d_b                (uchar*): Device array for color b (blue) pixels
// - d_bright_r         (uchar*): Device array for brightened r (red) pixels
// - d_bright_g         (uchar*): Device array for brightened g (green) pixels
// - d_bright_b         (uchar*): Device array for brightened b (blue) pixels
// - d_dark_r           (uchar*): Device array for darkened r (red) pixels
// - d_dark_g           (uchar*): Device array for darkened g (green) pixels
// - d_dark_b           (uchar*): Device array for darkened b (blue) pixels
// - d_grayscale        (uchar*): Device array for grayscaled pixels
// - brightPercentage   (int): Percentage to brighten the image
// - darkPercentage     (int): Percentage to darken the image
// - size               (int): Size of arrays
// Returns: None
__global__ void applyLightDarkGrayscale(uchar *d_r, uchar *d_g, uchar *d_b, uchar *d_bright_r, uchar *d_bright_g, uchar *d_bright_b,
                                        uchar *d_dark_r, uchar *d_dark_g, uchar *d_dark_b, uchar *d_grayscale, int brightPercentage,
                                        int darkPercentage, int size) {
    int blockId = blockIdx.x + blockIdx.y * gridDim.x
                    + gridDim.x * gridDim.y * blockIdx.z;
    int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
                            + (threadIdx.z * (blockDim.x * blockDim.y))
                            + (threadIdx.y * blockDim.x) + threadIdx.x;
    
    if (threadId < size) {
        // Bright
        d_bright_r[threadId] = brightenColor(d_r[threadId], brightPercentage);
        d_bright_g[threadId] = brightenColor(d_g[threadId], brightPercentage);
        d_bright_b[threadId] = brightenColor(d_b[threadId], brightPercentage);

        // Dark
        d_dark_r[threadId] = darkenColor(d_r[threadId], darkPercentage);
        d_dark_g[threadId] = darkenColor(d_g[threadId], darkPercentage);
        d_dark_b[threadId] = darkenColor(d_b[threadId], darkPercentage);

        // Grayscale
        d_grayscale[threadId] = (d_r[threadId] + d_g[threadId] + d_b[threadId]) / 3;
    }
}

// Sets up and executes the global kernel method
// Arguments:
// - ci                 (CudaImage*): CudaImage that will be modified in the kernel
// - brightPercentage   (int): Percentage to brighten the image
// - darkPercentage     (int): Percentage to darken the image
// Returns: None
__host__ void executeKernel(CudaImage *ci, int brightPercentage, int darkPercentage) {
    cout << "Running kernel...\n";

    const int blockZSize = 4;
    const int threadsPerBlock = 32;

    dim3 grid(ci->cols, ci->rows, 1);
    dim3 block(1, threadsPerBlock, blockZSize);

    // Kernel code
    int size = ci->rows * ci->cols;
    applyLightDarkGrayscale<<<grid, block>>>(ci->d_r, ci->d_g, ci->d_b, ci->d_bright_r, ci->d_bright_g, ci->d_bright_b,
                                            ci->d_dark_r, ci->d_dark_g, ci->d_dark_b, ci->d_grayscale, brightPercentage, darkPercentage, size);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch bright kernel: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    cout << "Done.\n";
}

// Parses the command line arguments accordingly and returns the appropriate values
// Arguments:
// - argc (int): Number of arguments
// - argv (char**): Array of char* (string), listing each argument
// Returns: tuple<int, int>
__host__ tuple<int, int> parseCliArguments(int argc, char **argv) {
    int brightPercentage = -1;
    int darkPercentage = -1;

    for (int i = 1; i+1 < argc; i+=2) {
        string mode = argv[i];
        if (mode.compare("-b") == 0 || mode.compare("--bright") == 0) {
            int temp = atoi(argv[i+1]);
            if (temp >= 0 && temp <= 100) {
                cout << "Setting bright percentage: " << temp << "%\n";
                brightPercentage = temp;
            } else {
                cout << "Error: Failed to read bright percentage.\n";
            }
        } else if (mode.compare("-d") == 0 || mode.compare("--dark") == 0) {
            int temp = atoi(argv[i+1]);
            if (temp >= 0 && temp <= 100) {
                cout << "Setting dark percentage: " << temp << "%\n";
                darkPercentage = temp;
            } else {
                cout << "Error: Failed to read dark percentage.\n";
            }
        }
    }

    return {brightPercentage, darkPercentage};
}

// Prompts the user for a percentage between 0 and 100, returning that value
// Arguments:
// - prompt (string): Text prompt that's shown to the user as they're prompted
// Returns: int
__host__ int promptInputPercentage(string prompt) {
    int response;
    do {
        cout << prompt << ": ";
        cin >> response;
    } while (response < 0 || response > 100);

    return response;
}

// Main function
// Arguments:
// - argc (int): Number of arguments
// - argv (char**): Array of char* (string), listing each argument
// Returns: int
__host__ int main(int argc, char **argv) {
    vector<string> filepaths;
    vector<CudaImage *> cudaImages;

    // Get a list of filenames
    string filepath = "./data";
    for (const auto &entry : filesystem::directory_iterator(filepath)) {
        string path = entry.path();
        if (path.rfind(".png") != string::npos) {
            filepaths.push_back(path);
        }
    }

    try {
        // Parse arguments, if any
        tuple<int, int> args = parseCliArguments(argc, argv);
        int brightPercentage = get<0>(args);
        int darkPercentage = get<1>(args);

        // Prompt user for percentage to brighten images (assuming no argument was passed)
        if (brightPercentage == -1) {
            brightPercentage = promptInputPercentage("Enter percentage to brighten first image");
            cout << "Setting bright percentage: " << brightPercentage << "%\n";
        }

        // Prompt user for percentage to darken images (assuming no argument was passed)
        if (darkPercentage == -1) {
            darkPercentage = promptInputPercentage("Enter percentage to darken second image");
            cout << "Setting bright percentage: " << darkPercentage << "%\n";
        }

        // Iterate through each file
        CudaImage *ci = NULL;
        for (string path : filepaths) {
            cout << "Processing file: " << path << "\n";
            ci = createCudaImage(path);
            copyFromHostToDevice(ci);

            // Execute kernel
            executeKernel(ci, brightPercentage, darkPercentage);

            cudaImages.push_back(ci);
        }

        cudaDeviceSynchronize();

        // Now that our data operations are finished, commence with mapping to output files
        for (int i = 0; i < cudaImages.size(); ++i) {
            copyFromDeviceToHost(cudaImages[i]);

            string filename = filepaths[i].substr(filepaths[i].find_last_of("/\\") + 1);
            mapBrightImage(cudaImages[i], "./data/output/bright/bright_" + filename);
            mapDarkImage(cudaImages[i], "./data/output/dark/dark_" + filename);
            mapGrayscaleImage(cudaImages[i], "./data/output/grayscale/grayscale_" + filename);

            destroyCudaImage(cudaImages[i]);
        }
        cleanUpDevice();

    } catch (Exception &e) {
        fprintf(stderr, "Caught exception: %s\n", e.what());
        exit(EXIT_FAILURE);
    }

    exit(EXIT_SUCCESS);
}