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

__device__ uchar brightenColor(uchar color, int percentage) {
    float modifier = 1 + percentage * 0.01;
    uchar temp = color;

    uchar newColor = color * modifier;
    if (newColor < temp) newColor = UCHAR_MAX; // Byte data wrapped around; set as MAX value

    return newColor;
}

__device__ uchar darkenColor(uchar color, int percentage) {
    float modifier = 1 - percentage * 0.01;
    return color * modifier;
}

__global__ void applyLightDarkGrayscale(uchar *d_r, uchar *d_g, uchar *d_b, uchar *d_bright_r, uchar *d_bright_g, uchar *d_bright_b,
                                        uchar *d_dark_r, uchar *d_dark_g, uchar *d_dark_b, uchar *d_grayscale, int brightPercentage,
                                        int darkPercentage, int size) {
    int blockId = blockIdx.x + blockIdx.y * gridDim.x
                    + gridDim.x * gridDim.y * blockIdx.z;
    int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
                            + (threadIdx.z * (blockDim.x * blockDim.y))
                            + (threadIdx.y * blockDim.x) + threadIdx.x;
    
    // if (threadId < size) {
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
    // }

    // __syncthreads();
}

__host__ void executeKernel(CudaImage *ci, int brightPercentage, int darkPercentage) {
    cout << "Running kernel...\n";

    const int blockZSize = 4;
    const int threadsPerBlock = 148;
    const int gridCols = min(ci->cols / (threadsPerBlock * 4), 1);

    printf("Grid cols: %d\n", gridCols);

    dim3 grid(ci->rows, gridCols, 1);
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

__host__ int promptInputPercentage(string prompt) {
    int response;
    do {
        cout << prompt << ": ";
        cin >> response;
    } while (response < 0 || response > 100);

    return response;
}

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
            executeKernel(ci, brightPercentage, darkPercentage);

            cudaImages.push_back(ci);
        }

        cudaDeviceSynchronize();

        cout << "Hello #1\n";

        // Now that our data operations are finished, commence with mapping to output files
        for (int i = 0; i < cudaImages.size(); ++i) {
            cout << "Hello #2\n";
            copyFromDeviceToHost(cudaImages[i]);

            // int size = cudaImages[i]->rows * cudaImages[i]->cols;
            // for (int j = 0; j < size; ++j) {
            //     printf("R: %d\n", ci->h_r[j]);
            //     printf("G: %d\n", ci->h_g[j]);
            //     printf("B: %d\n", ci->h_b[j]);
            // }

            cout << "Hello #3\n";
            string filename = filepaths[i].substr(filepaths[i].find_last_of("/\\") + 1);
            cout << "Hello #3\n";
            mapBrightImage(cudaImages[i], "./data/output/bright/bright_" + filename);
            cout << "Hello #3\n";
            mapDarkImage(cudaImages[i], "./data/output/dark/dark_" + filename);
            cout << "Hello #3\n";
            mapGrayscaleImage(cudaImages[i], "./data/output/grayscale/grayscale_" + filename);

            cout << "Hello #4\n";
            destroyCudaImage(cudaImages[i]);
        }
        cout << "Hello #5\n";
        cleanUpDevice();

    } catch (Exception &e) {
        fprintf(stderr, "Caught exception: %s\n", e.what());
        exit(EXIT_FAILURE);
    }

    exit(EXIT_SUCCESS);
}