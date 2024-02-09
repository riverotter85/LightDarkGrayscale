// File: CudaImage.hpp
// Author: Logan Davis
// Created: 1/24/2024
// Last Modified: 2/09/2024

#ifndef CUDA_IMAGE
#define CUDA_IMAGE

#include <limits>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace cv;
using namespace std;

// Takes the contents of an image and creates a brightened, darkened, and grayscale variant
// Example:
//      CudaImage *ci = createCudaImage("path/to/image.png");
//      copyFromHostToDevice(ci);
//
//      executeKernel(ci);
//
//      cudaDeviceSynchronize();
//
//      copyFromDeviceToHost(ci);
//      mapBrightImage(ci, "path/to/bright_image.png");
//      mapDarkImage(ci, "path/to/dark_image.png");
//      mapGrayscaleImage(ci, "path/to/grayscale_image.png");
//
//      destroyCudaImage(ci);
//      cleanUpDevice();
struct CudaImage {
    // Host pixels
    uchar *h_r;
    uchar *h_g;
    uchar *h_b;

    // Host bright pixels
    uchar *h_bright_r;
    uchar *h_bright_g;
    uchar *h_bright_b;

    // Host dark pixels
    uchar *h_dark_r;
    uchar *h_dark_g;
    uchar *h_dark_b;

    // Host grayscale pixels
    uchar *h_grayscale;

    // Device pixels
    uchar *d_r;
    uchar *d_g;
    uchar *d_b;

    // Device bright pixels
    uchar *d_bright_r;
    uchar *d_bright_g;
    uchar *d_bright_b;

    // Device dark pixels
    uchar *d_dark_r;
    uchar *d_dark_g;
    uchar *d_dark_b;

    // Device grayscale pixels
    uchar *d_grayscale;

    int rows;
    int cols;
};

// Background functions

// Reads image from file and retrieves number of rows, number of columns, and pixel values
// Arguments:
// - inputFile (string): Path to file that is read in
// Returns: tuple<int, int, uchar*, uchar*, uchar*>
__host__ tuple<int, int, uchar *, uchar *, uchar *> readImageFromFile(string inputFile) {
    Mat image = imread(inputFile, IMREAD_COLOR);

    const int rows = image.rows;
    const int cols = image.cols;

    uchar *h_r = (uchar *) malloc(sizeof(uchar) * rows * cols);
    uchar *h_g = (uchar *) malloc(sizeof(uchar) * rows * cols);
    uchar *h_b = (uchar *) malloc(sizeof(uchar) * rows * cols);

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

    return {rows, cols, h_r, h_g, h_b};
}

// Allocates unsigned character array on the device side
// Arguments:
// - size (size_t): Size of data to be allocated
// Returns: uchar*
__host__ uchar *allocateDeviceVector(size_t size) {
    uchar *vector = NULL;
    cudaError_t err = cudaMalloc(&vector, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    return vector;
}

// Deallocates unsigned character array on the device side
// Arguments:
// - vector (uchar*): Unsigned character array to be freed
// Returns: None
__host__ void deallocateDeviceVector(uchar *vector) {
    cudaError_t err = cudaFree(vector);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to free device vector: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// Allocates all necessary memory on the device side and returns the pointer values
// Arguments:
// - rows (int): Number of rows
// - cols (int): Number of columns
// Returns: tuple<uchar*, uchar*, uchar*, uchar*, uchar*, uchar*, uchar*, uchar*, uchar*, uchar*>
__host__ tuple<uchar *, uchar *, uchar *, uchar *, uchar *, uchar *, uchar *, uchar *, uchar *, uchar *> allocateDeviceMemory(int rows, int cols) {
    size_t size = rows * cols * sizeof(uchar);

    uchar *d_r         = allocateDeviceVector(size);
    uchar *d_g         = allocateDeviceVector(size);
    uchar *d_b         = allocateDeviceVector(size);

    uchar *d_bright_r  = allocateDeviceVector(size);
    uchar *d_bright_g  = allocateDeviceVector(size);
    uchar *d_bright_b  = allocateDeviceVector(size);

    uchar *d_dark_r    = allocateDeviceVector(size);
    uchar *d_dark_g    = allocateDeviceVector(size);
    uchar *d_dark_b    = allocateDeviceVector(size);

    uchar *d_grayscale = allocateDeviceVector(size);

    return {d_r, d_g, d_b, d_bright_r, d_bright_g, d_bright_b, d_dark_r, d_dark_g, d_dark_b, d_grayscale};
}

// Deallocates memory on the device side
// Arguments:
// - d_r         (uchar*): Device array of r (red) values
// - d_g         (uchar*): Device array of g (green) values
// - d_b         (uchar*): Device array of b (blue) values
// - d_bright_r  (uchar*): Device array of brightened r (red) values
// - d_bright_g  (uchar*): Device array of brightened g (green) values
// - d_bright_b  (uchar*): Device array of brightened b (blue) values
// - d_dark_r    (uchar*): Device array of darkened r (red) values
// - d_dark_g    (uchar*): Device array of darkened g (green) values
// - d_dark_b    (uchar*): Device array of darkened b (blue) values
// - d_grayscale (uchar*): Device array of grayscale pixel values
// Returns: None
__host__ void deallocateMemory(uchar *d_r, uchar *d_g, uchar *d_b, uchar *d_bright_r, uchar *d_bright_g, uchar *d_bright_b,
                                                    uchar *d_dark_r, uchar *d_dark_g, uchar *d_dark_b, uchar *d_grayscale) {
    deallocateDeviceVector(d_r);
    deallocateDeviceVector(d_g);
    deallocateDeviceVector(d_b);

    deallocateDeviceVector(d_bright_r);
    deallocateDeviceVector(d_bright_g);
    deallocateDeviceVector(d_bright_b);

    deallocateDeviceVector(d_dark_r);
    deallocateDeviceVector(d_dark_g);
    deallocateDeviceVector(d_dark_b);

    deallocateDeviceVector(d_grayscale);
}

// Copies data from host to the device using CUDA
// Arguments:
// - src (uchar): Source data being copied
// - dst (uchar): Destination that we're copying to
// - size (size_t): Size of data
// Returns: None
__host__ void copyHostDevice(uchar *src, uchar *dst, size_t size) {
    cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy from host to device: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// Copies data from device to the host using CUDA
// Arguments:
// - src (uchar): Source data being copied
// - dst (uchar): Destination that we're copying to
// - size (size_t): Size of data
// Returns: None
__host__ void copyDeviceHost(uchar *src, uchar *dst, size_t size) {
    cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy from device to host: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// Maps image using the provided pixel values, writing them into a new image file
// Arguments:
// - filter_r   (uchar*): Array of filtered r (red) pixel values
// - filter_g   (uchar*): Array of filtered g (green) pixel values
// - filter_b   (uchar*): Array of filtered b (blue) pixel values
// - rows       (int): Number of image rows
// - cols       (int): Number of image columns
// - outputFile (string): Path to new file that is created
// Returns: None
__host__ void mapImage(uchar *filter_r, uchar *filter_g, uchar *filter_b, int rows, int cols, string outputFile) {
    // Create Mat of type unsigned char with 3 channels
    Mat imageMat(rows, cols, CV_8UC3);
    vector<int> compressionParams;
    compressionParams.push_back(IMWRITE_PNG_COMPRESSION);
    compressionParams.push_back(9);

    // Copy the pixel values to map out the new image
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            int index = row * cols + col;
            Vec3b color = imageMat.at<Vec3b>(Point(col, row));
            color.val[0] = filter_b[index];
            color.val[1] = filter_g[index];
            color.val[2] = filter_r[index];
            imageMat.at<Vec3b>(Point(col, row)) = color;
        }
    }

    imwrite(outputFile, imageMat, compressionParams);
}

// Main controls

// Creates new CudaImage from image filepath
// Arguments:
// - inputImage (string): Path to image file that is read in
// Returns: CudaImage*
__host__ CudaImage *createCudaImage(string inputImage) {
    // Allocate host pixel values
    auto[rows, cols, h_r, h_g, h_b] = readImageFromFile(inputImage);

    uchar *h_bright_r  = (uchar *) malloc(sizeof(uchar) * rows * cols);
    uchar *h_bright_g  = (uchar *) malloc(sizeof(uchar) * rows * cols);
    uchar *h_bright_b  = (uchar *) malloc(sizeof(uchar) * rows * cols);

    uchar *h_dark_r    = (uchar *) malloc(sizeof(uchar) * rows * cols);
    uchar *h_dark_g    = (uchar *) malloc(sizeof(uchar) * rows * cols);
    uchar *h_dark_b    = (uchar *) malloc(sizeof(uchar) * rows * cols);

    uchar *h_grayscale = (uchar *) malloc(sizeof(uchar) * rows * cols);

    // Allocate device pixel values
    tuple<uchar *, uchar *, uchar *, uchar *, uchar *, uchar *, uchar *, uchar *, uchar *, uchar *> deviceTuple = allocateDeviceMemory(rows, cols);
    uchar *d_r         = get<0>(deviceTuple);
    uchar *d_g         = get<1>(deviceTuple);
    uchar *d_b         = get<2>(deviceTuple);
    uchar *d_bright_r  = get<3>(deviceTuple);
    uchar *d_bright_g  = get<4>(deviceTuple);
    uchar *d_bright_b  = get<5>(deviceTuple);
    uchar *d_dark_r    = get<6>(deviceTuple);
    uchar *d_dark_g    = get<7>(deviceTuple);
    uchar *d_dark_b    = get<8>(deviceTuple);
    uchar *d_grayscale = get<9>(deviceTuple);

    // Set properties in new CudaImage project
    CudaImage *cudaImage = (CudaImage *) malloc(sizeof(CudaImage));

    cudaImage->h_r = h_r;
    cudaImage->h_g = h_g;
    cudaImage->h_b = h_b;

    cudaImage->h_bright_r  = h_bright_r;
    cudaImage->h_bright_g  = h_bright_g;
    cudaImage->h_bright_b  = h_bright_b;

    cudaImage->h_dark_r    = h_dark_r;
    cudaImage->h_dark_g    = h_dark_g;
    cudaImage->h_dark_b    = h_dark_b;

    cudaImage->h_grayscale = h_grayscale;

    cudaImage->d_r = d_r;
    cudaImage->d_g = d_g;
    cudaImage->d_b = d_b;

    cudaImage->d_bright_r  = d_bright_r;
    cudaImage->d_bright_g  = d_bright_g;
    cudaImage->d_bright_b  = d_bright_b;

    cudaImage->d_dark_r    = d_dark_r;
    cudaImage->d_dark_g    = d_dark_g;
    cudaImage->d_dark_b    = d_dark_b;

    cudaImage->d_grayscale = d_grayscale;

    cudaImage->rows = rows;
    cudaImage->cols = cols;

    return cudaImage;
}

// Destroys CudaImage
// Arguments:
// - cudaImage (CudaImage*): CudaImage to be destroyed
// Returns: None
__host__ void destroyCudaImage(CudaImage *cudaImage) {
    deallocateMemory(cudaImage->d_r, cudaImage->d_g, cudaImage->d_b,
                    cudaImage->d_bright_r, cudaImage->d_bright_g, cudaImage->d_bright_b,
                    cudaImage->d_dark_r, cudaImage->d_dark_g, cudaImage->d_dark_b, cudaImage->d_grayscale);
}

// Cleans up device data using CUDA methods
// Arguments: None
// Returns: None
__host__ void cleanUpDevice() {
    cudaError_t err = cudaDeviceReset();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to reset the device state: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// Copies from host to device using delegated methods
// Arguments:
// - cudaImage (CudaImage*): CudaImage that is being copied
// Returns: None
__host__ void copyFromHostToDevice(CudaImage *cudaImage) {
    size_t size = cudaImage->rows * cudaImage->cols * sizeof(uchar);

    copyHostDevice(cudaImage->h_r, cudaImage->d_r, size);
    copyHostDevice(cudaImage->h_g, cudaImage->d_g, size);
    copyHostDevice(cudaImage->h_b, cudaImage->d_b, size);

    copyHostDevice(cudaImage->h_bright_r, cudaImage->d_bright_r, size);
    copyHostDevice(cudaImage->h_bright_g, cudaImage->d_bright_g, size);
    copyHostDevice(cudaImage->h_bright_b, cudaImage->d_bright_b, size);

    copyHostDevice(cudaImage->h_dark_r, cudaImage->d_dark_r, size);
    copyHostDevice(cudaImage->h_dark_g, cudaImage->d_dark_g, size);
    copyHostDevice(cudaImage->h_dark_b, cudaImage->d_dark_b, size);

    copyHostDevice(cudaImage->h_grayscale, cudaImage->d_grayscale, size);
}

// Copies from device to host using delegated methods
// Arguments:
// - cudaImage (CudaImage*): CudaImage that is being copied
// Returns: None
__host__ void copyFromDeviceToHost(CudaImage *cudaImage) {
    size_t size = cudaImage->rows * cudaImage->cols * sizeof(uchar);

    copyDeviceHost(cudaImage->d_r, cudaImage->h_r, size);
    copyDeviceHost(cudaImage->d_g, cudaImage->h_g, size);
    copyDeviceHost(cudaImage->d_b, cudaImage->h_b, size);

    copyDeviceHost(cudaImage->d_bright_r, cudaImage->h_bright_r, size);
    copyDeviceHost(cudaImage->d_bright_g, cudaImage->h_bright_g, size);
    copyDeviceHost(cudaImage->d_bright_b, cudaImage->h_bright_b, size);

    copyDeviceHost(cudaImage->d_dark_r, cudaImage->h_dark_r, size);
    copyDeviceHost(cudaImage->d_dark_g, cudaImage->h_dark_g, size);
    copyDeviceHost(cudaImage->d_dark_b, cudaImage->h_dark_b, size);

    copyDeviceHost(cudaImage->d_grayscale, cudaImage->h_grayscale, size);
}

// Maps brightened image using delegated mapImage method
// Arguments:
// - cudaImage (CudaImage*): CudaImage that is being mapped
// - outputFile (string): Path to file that is being created
// Returns: None
__host__ void mapBrightImage(CudaImage *cudaImage, string outputFile) {
    mapImage(cudaImage->h_bright_r, cudaImage->h_bright_g, cudaImage->h_bright_b, cudaImage->rows, cudaImage->cols, outputFile);
}

// Maps darkened image using delegated mapImage method
// Arguments:
// - cudaImage (CudaImage*): CudaImage that is being mapped
// - outputFile (string): Path to file that is being created
// Returns: None
__host__ void mapDarkImage(CudaImage *cudaImage, string outputFile) {
    mapImage(cudaImage->h_dark_r, cudaImage->h_dark_g, cudaImage->h_dark_b, cudaImage->rows, cudaImage->cols, outputFile);
}

// Maps grayscaled image using delegated mapImage method
// Arguments:
// - cudaImage (CudaImage*): CudaImage that is being mapped
// - outputFile (string): Path to file that is being created
// Returns: None
__host__ void mapGrayscaleImage(CudaImage *cudaImage, string outputFile) {
    mapImage(cudaImage->h_grayscale, cudaImage->h_grayscale, cudaImage->h_grayscale, cudaImage->rows, cudaImage->cols, outputFile);
}

#endif