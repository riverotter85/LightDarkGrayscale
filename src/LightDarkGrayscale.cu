#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>
#include <list>
#include <exception>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include "CudaImage.hpp"

using namespace std;

__global__ void bright(uchar *d_r, uchar *d_g, uchar *d_b, int percentage, uchar *d_bright) {
    int blockId = blockIdx.x + blockIdx.y * gridDim.x
                    + gridDim.x * gridDim.y * blockIdx.z;
    int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
                            + (threadIdx.z * (blockDim.x * blockDim.y))
                            + (threadIdx.y * blockDim.x) + threadIdx.x;
    
    float modifier = 1 + percentage * 0.01;
    d_bright[threadId] = (d_r[threadId] + d_g[threadId] + d_b[threadId]) * modifier;
}

__global__ void dark(uchar *d_r, uchar *d_g, uchar *d_b, int percentage, uchar *d_dark) {
    int blockId = blockIdx.x + blockIdx.y * gridDim.x
                    + gridDim.x * gridDim.y * blockIdx.z;
    int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
                            + (threadIdx.z * (blockDim.x * blockDim.y))
                            + (threadIdx.y * blockDim.x) + threadIdx.x;
    
    float modifier = 1 - percentage * 0.01;
    d_dark[threadId] = (d_r[threadId] + d_g[threadId] + d_b[threadId]) * modifier;
}

__global__ void grayscale(uchar *d_r, uchar *d_g, uchar *d_b, uchar *d_grayscale) {
    int blockId = blockIdx.x + blockIdx.y * gridDim.x
                    + gridDim.x * gridDim.y * blockIdx.z;
    int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
                            + (threadIdx.z * (blockDim.x * blockDim.y))
                            + (threadIdx.y * blockDim.x) + threadIdx.x;
    
    d_grayscale[threadId] = (d_r[threadId] + d_g[threadId] + d_b[threadId]) / 3;
}

__host__ void executeKernel(CudaImage *ci, int brightPercentage, int darkPercentage, int threadsPerBlock) {
    cout << "Running kernel for LightDarkGrayscale...\n";

    const int blockZSize = 4;
    const int gridCols = min(ci->cols / (threadsPerBlock * 4), 1);
    dim3 grid(ci->rows, gridCols, 1);
    dim3 block(1, threadsPerBlock, blockZSize);

    // Bright kernel
    bright<<<grid, block>>>(ci->d_r, ci->d_g, ci->d_b, brightPercentage, ci->d_bright);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch bright kernel: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Dark kernel
    dark<<<grid, block>>>(ci->d_r, ci->d_g, ci->d_b, darkPercentage, ci->d_dark);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch dark kernel: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Grayscale kernel
    grayscale<<<grid, block>>>(ci->d_r, ci->d_g, ci->d_b, ci->d_grayscale);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch grayscale kernel: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    cout << "Done.\n";
}

__host__ int promptInputPercentage(string prompt) {
    int response;
    do {
        cout << prompt << ": ";
        cin >> response;
    } while (response < 0 || response > 100);

    return response;
}

__host__ int main(int argc, char **argv) {
    const int threadsPerBlock = 8;
    string filepaths;

    // Get a list of filenames
    string filepath = "./data/input";
    for (const auto &entry : filesystem::directory_iterator(filepath)) {
        filepaths.push_back(entry.path());
    }

    try {
        // Prompt user for percentage to brighten images by
        int brightPercentage = promptInputPercentage("Enter percentage to brighten first image");

        // Prompt user for percentage to darken images by
        int darkPercentage = promptInputPercentage("Enter percentage to darken second image");

        // Iterate through each file
        for (string path : filepaths) {
            CudaImage ci = createCudaImage(path);
            copyFromHostToDevice(ci);

            // Execute kernel
            executeKernel(ci, brightPercentage, darkPercentage, threadsPerBlock);
        }

        // Synchronize all threads
        __synchronize_threads();

        // Now that our data operations are finished, commence with mapping to output files
        for (string path : filepaths) {
            copyFromDeviceToHost(ci);

            mapBrightImage("bright_" + path);
            mapDarkImage("dark_" + path);
            mapGrayscaleImage("grayscale_" + path);
        }

    } catch (Exception &e) {
        fprintf(stderr, "Caught exception: %s\n", e.what());
        exit(EXIT_FAILURE);
    }

    exit(EXIT_SUCCESS);
}