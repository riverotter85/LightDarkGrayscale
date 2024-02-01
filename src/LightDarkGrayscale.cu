#include <iostream>
#include <fstream>
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

__global__ void applyLightDarkGrayscale(uchar *d_r, uchar *d_g, uchar *d_b, uchar *d_bright, uchar *d_dark, uchar *d_grayscale, int brightPercentage, int darkPercentage) {
    int blockId = blockIdx.x + blockIdx.y * gridDim.x
                    + gridDim.x * gridDim.y * blockIdx.z;
    int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
                            + (threadIdx.z * (blockDim.x * blockDim.y))
                            + (threadIdx.y * blockDim.x) + threadIdx.x;
    
    // Bright
    float modifier = 1 + brightPercentage * 0.01;
    d_bright[threadId] = (d_r[threadId] + d_g[threadId] + d_b[threadId]) * modifier;

    // Dark
    modifier = 1 - darkPercentage * 0.01;
    d_dark[threadId] = (d_r[threadId] + d_g[threadId] + d_b[threadId]) * modifier;

    // Grayscale
    d_grayscale[threadId] = (d_r[threadId] + d_g[threadId] + d_b[threadId]) / 3;

    // __syncthreads();
}

__host__ void executeKernel(CudaImage *ci, int brightPercentage, int darkPercentage, dim3 threadsPerBlock, dim3 blocksPerGrid) {
    cout << "Running kernel for LightDarkGrayscale...\n";

    // const int blockZSize = 4;
    // const int gridCols = min(ci->cols / (threadsPerBlock * 4), 1);
    // dim3 grid(ci->rows, gridCols, 1);
    // dim3 block(1, threadsPerBlock, blockZSize);

    // printf("Grid: {%d, %d, %d} blocks. Blocks: ${%d, %d, %d} threads.\n", grid.x, grid.y, grid.z, block.x, block.y, block.z);

    // Kernel code
    applyLightDarkGrayscale<<<blocksPerGrid, threadsPerBlock>>>(ci->d_r, ci->d_g, ci->d_b, ci->d_bright, ci->d_dark, ci->d_grayscale, brightPercentage, darkPercentage);
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
        if (path.rfind(".gitkeep") == string::npos) {
            filepaths.push_back(path);
            cout << "Filepath added: " << path << "\n";
        }
    }

    try {
        // Prompt user for percentage to brighten images by
        int brightPercentage = promptInputPercentage("Enter percentage to brighten first image");

        // Prompt user for percentage to darken images by
        int darkPercentage = promptInputPercentage("Enter percentage to darken second image");

        cout << "Before start loop\n";

        // Iterate through each file
        CudaImage *ci = NULL;
        for (string path : filepaths) {
            cout << "Size: " << cudaImages.size() << "\n";
            cout << "Path: " << path << "\n";
            ci = createCudaImage(path);
            copyFromHostToDevice(ci);

            cout << "Inside loop #1\n";

            // Execute kernel
            executeKernel(ci, brightPercentage, darkPercentage, threadsPerBlock, blocksPerGrid);

            cout << "Inside loop #2\n";

            cudaImages.push_back(ci);

            cout << "Inside loop #3\n";
        }

        // cout << "Before end loop\n";
        cudaDeviceSynchronize();

        // Now that our data operations are finished, commence with mapping to output files
        for (int i = 0; i < filepaths.size(); ++i) {
            cout << "In end loop #1\n";
            copyFromDeviceToHost(cudaImages[i]);

            cout << "In end loop #2\n";

            mapBrightImage(cudaImages[i], "bright_" + filepaths[i]);
            mapDarkImage(cudaImages[i], "dark_" + filepaths[i]);
            mapGrayscaleImage(cudaImages[i], "grayscale_" + filepaths[i]);

            cout << "Hello\n";

            destroyCudaImage(cudaImages[i]);
        }

    } catch (Exception &e) {
        fprintf(stderr, "Caught exception: %s\n", e.what());
        exit(EXIT_FAILURE);
    }

    exit(EXIT_SUCCESS);
}