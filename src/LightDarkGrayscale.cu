#include <iostream>
#include <string>
#include <filesystem>
#include <list>
#include <exception>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include "CudaImage.hpp"

using namespace std;

__global__ void applyLightDarkGrayscale(uchar *d_r, uchar *d_g, uchar *d_b, uchar *d_bright, uchar *d_dark, uchar *d_grayscale, int brightPercentage, int darkPercentage, int size) {
    int blockId = blockIdx.x + blockIdx.y * gridDim.x
                    + gridDim.x * gridDim.y * blockIdx.z;
    int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
                            + (threadIdx.z * (blockDim.x * blockDim.y))
                            + (threadIdx.y * blockDim.x) + threadIdx.x;
    
    if (threadId < size) {
        // Bright
        float modifier = 1 + brightPercentage * 0.01;
        d_bright[threadId] = (d_r[threadId] + d_g[threadId] + d_b[threadId]) * modifier;

        // Dark
        modifier = 1 - darkPercentage * 0.01;
        d_dark[threadId] = (d_r[threadId] + d_g[threadId] + d_b[threadId]) * modifier;

        // Grayscale
        d_grayscale[threadId] = (d_r[threadId] + d_g[threadId] + d_b[threadId]) / 3;
    }

    // __syncthreads();
}

__host__ void executeKernel(CudaImage *ci, int brightPercentage, int darkPercentage, dim3 threadsPerBlock, dim3 blocksPerGrid) {
    cout << "Running kernel...\n";

    // Kernel code
    int size = ci->rows * ci->cols;
    applyLightDarkGrayscale<<<blocksPerGrid, threadsPerBlock>>>(ci->d_r, ci->d_g, ci->d_b, ci->d_bright, ci->d_dark, ci->d_grayscale, brightPercentage, darkPercentage, size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch bright kernel: %s\n", cudaGetErrorString(err));
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
    // const int threadsPerBlock = 8;
    dim3 threadsPerBlock;
    dim3 blocksPerGrid;

    threadsPerBlock.x = 32;
    threadsPerBlock.y = 1;
    threadsPerBlock.z = 1;

    blocksPerGrid.x = 8;
    blocksPerGrid.y = 1;
    blocksPerGrid.z = 1;

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
        // Prompt user for percentage to brighten images
        int brightPercentage = promptInputPercentage("Enter percentage to brighten first image");

        // Prompt user for percentage to darken images
        int darkPercentage = promptInputPercentage("Enter percentage to darken second image");

        // Iterate through each file
        CudaImage *ci = NULL;
        for (string path : filepaths) {
            ci = createCudaImage(path);
            copyFromHostToDevice(ci);

            // Execute kernel
            executeKernel(ci, brightPercentage, darkPercentage, threadsPerBlock, blocksPerGrid);

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