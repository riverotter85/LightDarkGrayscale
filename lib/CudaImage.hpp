#ifndef CUDA_IMAGE
#define CUDA_IMAGE

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <cxxabi.h> # NOTE: Remove!!

using namespace cv;
using namespace std;

struct CudaImage {
    uchar *h_r;
    uchar *h_g;
    uchar *h_b;

    uchar *h_bright;
    uchar *h_dark;
    uchar *h_grayscale;

    uchar *d_r;
    uchar *d_g;
    uchar *d_b;

    uchar *d_bright;
    uchar *d_dark;
    uchar *d_grayscale;

    int rows;
    int cols;
};

// Background functions

__host__ tuple<int, int, uchar *, uchar *, uchar *> readImageFromFile(string inputFile) {
    Mat image = imread(inputFile, IMREAD_COLOR);

    const int rows = image.rows;
    const int cols = image.cols;

    uchar *h_r = (uchar *) malloc(sizeof(uchar) * rows * cols);
    uchar *h_g = (uchar *) malloc(sizeof(uchar) * rows * cols);
    uchar *h_b = (uchar *) malloc(sizeof(uchar) * rows * cols);

    cout << "Before Hello\n";

    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            Vec3b intensity = image.at<Vec3b>(row, col);
            uchar b = intensity.val[0];
            uchar g = intensity.val[1];
            uchar r = intensity.val[2];

            h_r[row * cols + col] = r;
            h_g[row * cols + col] = g;
            h_b[row * cols + col] = b;
        }
    }

    cout << "Hello\n";

    return {rows, cols, h_r, h_g, h_b};
}

__host__ uchar *allocateDeviceVector(size_t size) {
    uchar *vector = NULL;
    cudaError_t err = cudaMalloc(&vector, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    return vector;
}

__host__ void deallocateDeviceVector(uchar *vector) {
    cudaError_t err = cudaFree(vector);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to free device vector: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

__host__ tuple<uchar *, uchar *, uchar *, uchar *, uchar *, uchar *> allocateDeviceMemory(int rows, int cols) {
    size_t size = rows * cols * sizeof(uchar);

    uchar *d_r         = allocateDeviceVector(size);
    uchar *d_g         = allocateDeviceVector(size);
    uchar *d_b         = allocateDeviceVector(size);
    uchar *d_bright    = allocateDeviceVector(size);
    uchar *d_dark      = allocateDeviceVector(size);
    uchar *d_grayscale = allocateDeviceVector(size);

    return {d_r, d_g, d_b, d_bright, d_dark, d_grayscale};
}

__host__ void deallocateMemory(uchar *d_r, uchar *d_g, uchar *d_b, uchar *d_bright, uchar *d_dark, uchar *d_grayscale) {
    deallocateDeviceVector(d_r);
    deallocateDeviceVector(d_g);
    deallocateDeviceVector(d_b);
    deallocateDeviceVector(d_bright);
    deallocateDeviceVector(d_dark);
    deallocateDeviceVector(d_grayscale);
}

__host__ void cleanUpDevice() {
    cudaError_t err = cudaDeviceReset();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to reset the device state: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

__host__ void copyHostDevice(uchar *src, uchar *dst, size_t size) {
    cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy from host to device: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

__host__ void copyDeviceHost(uchar *src, uchar *dst, size_t size) {
    cout << "Hello!\n";
    cout << "Src Type: " << abi::__cxa_demangle(typeid(src).name(), 0, 0, 0) << "\n";
    cout << "Dst Type: " << abi::__cxa_demangle(typeid(dst).name(), 0, 0, 0) << "\n";
    cout << "Size Type: " << abi::__cxa_demangle(typeid(size).name(), 0, 0, 0) << "\n";
    cout << "Copy Type: " << abi::__cxa_demangle(typeid(cudaMemcpyDeviceToHost).name(), 0, 0, 0) << "\n";
    cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy from device to host: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

__host__ void mapImage(uchar *filter, int rows, int cols, string outputFile) {
    Mat imageMat(rows, cols, CV_8UC1);
    vector<int> compressionParams;
    compressionParams.push_back(IMWRITE_PNG_COMPRESSION);
    compressionParams.push_back(9);

    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            imageMat.at<uchar>(row, col) = filter[row * rows + cols];
        }
    }

    imwrite(outputFile, imageMat, compressionParams);
}

// Main controls

__host__ CudaImage *createCudaImage(string inputImage) {
    auto[rows, cols, h_r, h_g, h_b] = readImageFromFile(inputImage);
    uchar *h_bright    = (uchar *) malloc(sizeof(uchar) * rows * cols);
    uchar *h_dark      = (uchar *) malloc(sizeof(uchar) * rows * cols);
    uchar *h_grayscale = (uchar *) malloc(sizeof(uchar) * rows * cols);

    tuple<uchar *, uchar *, uchar *, uchar *, uchar *, uchar *> deviceTuple = allocateDeviceMemory(rows, cols);
    uchar *d_r         = get<0>(deviceTuple);
    uchar *d_g         = get<1>(deviceTuple);
    uchar *d_b         = get<2>(deviceTuple);
    uchar *d_bright    = get<3>(deviceTuple);
    uchar *d_dark      = get<4>(deviceTuple);
    uchar *d_grayscale = get<5>(deviceTuple);

    // Set properties in new CudaImage project
    CudaImage *cudaImage = (CudaImage *) malloc(sizeof(CudaImage));

    cudaImage->h_r = h_r;
    cudaImage->h_g = h_g;
    cudaImage->h_b = h_b;

    cudaImage->h_bright    = h_bright;
    cudaImage->h_dark      = h_dark;
    cudaImage->h_grayscale = h_grayscale;

    cudaImage->d_r = d_r;
    cudaImage->d_g = d_g;
    cudaImage->d_b = d_b;

    cudaImage->d_bright    = d_bright;
    cudaImage->d_dark      = d_dark;
    cudaImage->d_grayscale = d_grayscale;

    cudaImage->rows = rows;
    cudaImage->cols = cols;

    return cudaImage;
}

__host__ void toString(CudaImage *cudaImage) {
    
}

__host__ void destroyCudaImage(CudaImage *cudaImage) {
    deallocateMemory(cudaImage->d_r, cudaImage->d_g, cudaImage->d_b,
                    cudaImage->d_bright, cudaImage->d_dark, cudaImage->d_grayscale);
    cleanUpDevice();
}

__host__ void copyFromHostToDevice(CudaImage *cudaImage) {
    size_t size = cudaImage->rows * cudaImage->cols * sizeof(uchar);

    copyHostDevice(cudaImage->h_r, cudaImage->d_r, size);
    copyHostDevice(cudaImage->h_g, cudaImage->d_g, size);
    copyHostDevice(cudaImage->h_b, cudaImage->d_b, size);
}

__host__ void copyFromDeviceToHost(CudaImage *cudaImage) {
    size_t size = cudaImage->rows * cudaImage->cols * sizeof(uchar);

    printf("D: %p\n", cudaImage->d_r);
    printf("D: %p\n", cudaImage->d_g);
    printf("D: %p\n", cudaImage->d_b);
    printf("H: %p\n", cudaImage->h_r);
    printf("H: %p\n", cudaImage->h_g);
    printf("H: %p\n", cudaImage->h_b);
    copyDeviceHost(cudaImage->d_r, cudaImage->h_r, size);
    copyDeviceHost(cudaImage->d_g, cudaImage->h_g, size);
    copyDeviceHost(cudaImage->d_b, cudaImage->h_b, size);
}

__host__ void mapBrightImage(CudaImage *cudaImage, string outputFile) {
    mapImage(cudaImage->h_bright, cudaImage->rows, cudaImage->cols, outputFile);
}

__host__ void mapDarkImage(CudaImage *cudaImage, string outputFile) {
    mapImage(cudaImage->h_dark, cudaImage->rows, cudaImage->cols, outputFile);
}

__host__ void mapGrayscaleImage(CudaImage *cudaImage, string outputFile) {
    mapImage(cudaImage->h_grayscale, cudaImage->rows, cudaImage->cols, outputFile);
}

#endif