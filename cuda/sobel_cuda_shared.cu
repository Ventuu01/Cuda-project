#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <filesystem>
#include <chrono>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// =================== LOAD/SAVE  ===================

bool loadPPM(const std::string& path, int& width, int& height, int& maxVal,
             std::vector<unsigned char>& rgb)
{
    std::ifstream in(path);
    if (!in.is_open()) {
        std::cerr << "Error opening PPM file: " << path << std::endl;
        return false;
    }

    std::string type;
    in >> type;
    if (type != "P3") {
        std::cerr << "Unsupported PPM type, expected P3: " << type << std::endl;
        return false;
    }

     char c = in.peek();
    while (c == '#') {
        std::string line;
        std::getline(in, line);
        c = in.peek();
    }

    in >> width >> height >> maxVal;
    if (!in.good()) {
        std::cerr << "Error parsing PPM header\n";
        return false;
    }

    rgb.resize(width * height * 3);
    for (int i = 0; i < width * height * 3; i++) {
        int val;
        in >> val;
        if (!in.good()) {
            std::cerr << "Error reading pixel data\n";
            return false;
        }
        rgb[i] = static_cast<unsigned char>(val);
    }

    return true;
}

bool savePPM(const std::string& path, int width, int height, int maxVal,
             const std::vector<unsigned char>& rgb)
{
    std::ofstream out(path);
    if (!out.is_open()) {
        std::cerr << "Error opening output file: " << path << std::endl;
        return false;
    }

    out << "P3\n";
    out << width << " " << height << "\n";
    out << maxVal << "\n";

    for (int i = 0; i < width * height; i++) {
        int r = rgb[3 * i + 0];
        int g = rgb[3 * i + 1];
        int b = rgb[3 * i + 2];
        out << r << " " << g << " " << b << "\n";
    }

    return true;
}

// =================== KERNEL CON SHARED MEMORY ===================

const int TILE_W = 16;
const int TILE_H = 16;

// Kernel Sobel con tiling in shared memory (tile + halo di 1 pixel)
__global__ void sobelSharedKernel(const unsigned char* inputGray,
                                  unsigned char* outputGray,
                                  int width, int height)
{
    __shared__ unsigned char tile[TILE_H + 2][TILE_W + 2];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int x = blockIdx.x * TILE_W + tx;
    int y = blockIdx.y * TILE_H + ty;

    // index nella shared 
    int sx = tx + 1;
    int sy = ty + 1;

    // Carica il pixel centrale 
    if (x < width && y < height) {
        tile[sy][sx] = inputGray[y * width + x];
    } else {
        tile[sy][sx] = 0;
    }

    // Carica halo a sinistra
    if (tx == 0 && x > 0 && y < height) {
        tile[sy][0] = inputGray[y * width + (x - 1)];
    }
    // halo a destra
    if (tx == blockDim.x - 1 && x < width - 1 && y < height) {
        tile[sy][TILE_W + 1] = inputGray[y * width + (x + 1)];
    }
    // halo in alto
    if (ty == 0 && y > 0 && x < width) {
        tile[0][sx] = inputGray[(y - 1) * width + x];
    }
    // halo in basso
    if (ty == blockDim.y - 1 && y < height - 1 && x < width) {
        tile[TILE_H + 1][sx] = inputGray[(y + 1) * width + x];
    }

    // angolo alto-sinistra
    if (tx == 0 && ty == 0 && x > 0 && y > 0) {
        tile[0][0] = inputGray[(y - 1) * width + (x - 1)];
    }
    // angolo alto-destra
    if (tx == blockDim.x - 1 && ty == 0 && x < width - 1 && y > 0) {
        tile[0][TILE_W + 1] = inputGray[(y - 1) * width + (x + 1)];
    }
    // angolo basso-sinistra
    if (tx == 0 && ty == blockDim.y - 1 && x > 0 && y < height - 1) {
        tile[TILE_H + 1][0] = inputGray[(y + 1) * width + (x - 1)];
    }
    // angolo basso-destra
    if (tx == blockDim.x - 1 && ty == blockDim.y - 1 &&
        x < width - 1 && y < height - 1) {
        tile[TILE_H + 1][TILE_W + 1] = inputGray[(y + 1) * width + (x + 1)];
    }

    __syncthreads();

    // Bordo immagine --> salta i pixel fuori
    if (x <= 0 || x >= width - 1 || y <= 0 || y >= height - 1)
        return;

    // Kernel Sobel
    int Gx[3][3] = {
        { 1,  0, -1},
        { 2,  0, -2},
        { 1,  0, -1}
    };

    int Gy[3][3] = {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };

    int sumX = 0;
    int sumY = 0;

    // Usa i valori dalla shared memory 
    for (int ky = -1; ky <= 1; ky++) {
        for (int kx = -1; kx <= 1; kx++) {
            int val = tile[sy + ky][sx + kx];
            sumX += val * Gx[ky + 1][kx + 1];
            sumY += val * Gy[ky + 1][kx + 1];
        }
    }

    float mag = sqrtf(float(sumX * sumX + sumY * sumY));
    unsigned char g = (mag >= 100.0f) ? 255 : 0;

    int idx = y * width + x;
    outputGray[idx] = g;
}
// BLUE ilter kernel
__global__ void blueKernel(const unsigned char* inRGB,
                                unsigned char* outRGB,
                                int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = (y * width + x) * 3;

    unsigned char r = inRGB[idx + 0];
    unsigned char g = inRGB[idx + 1];
    unsigned char b = inRGB[idx + 2];

    int bb = (int)b + 100;
    if (bb >= 256) bb = 255; 

    outRGB[idx + 0] = r;
    outRGB[idx + 1] = g;
    outRGB[idx + 2] = (unsigned char)bb;
}


// =================== MAIN ===================


namespace fs = std::filesystem;

int main()
{
    // fs::create_directories("../data/output/logs/");
    // // SAVE logs into txt file
    // std::ofstream logFile("../data/output/logs/cuda_shared_output.txt");
    // std::streambuf* originalCout = std::cout.rdbuf();

    // if (logFile.is_open()) {
    //     std::cout.rdbuf(logFile.rdbuf());
    // }
    std::string inputDir  = "../data/input/dataset/ppm/";
    // std::string inputDir  = "../data/input/dataset_1/";
    std::string outputDir = "../data/output/sobel_shared/";
    std::string sobelDir   = outputDir + "sobel/";
    std::string blueDir    = outputDir + "blue/";
    fs::create_directories(sobelDir);
    fs::create_directories(blueDir);

    std::cout << "CUDA BENCHMARK\n";
    std::cout << "Dataset: " << inputDir << "\n";
    std::cout << "Output folder: " << outputDir << "\n";
    std::cout << "Block size: " << TILE_W << "x" << TILE_H << " Sobel\n";


    // Greyscale sobel buffers
    unsigned char *d_in = nullptr, *d_out = nullptr;
    size_t allocatedBytes = 0;

    // Blue filters buffers
    unsigned char *d_rgb_in = nullptr, *d_rgb_out = nullptr;
    size_t allocatedRgbBytes = 0;

    // Eventi
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int imageCount = 0;
    // double totalKernelMs = 0.0;
    double totalSobelKernelMs = 0.0;
    double totalBlueKernelMs  = 0.0;


    auto totalStart = std::chrono::high_resolution_clock::now();

    // LOOP iimmagine
    for (const auto& entry : fs::directory_iterator(inputDir))
    {
        if (entry.path().extension() != ".ppm")
            continue;

        std::string inputPath = entry.path().string();
        std::string name      = entry.path().stem().string();
        std::string outputPath = sobelDir + name + "_sobel_cuda_shared.ppm";
        std::string bluePath   = blueDir + name + "_blue_cuda.ppm";

        int width = 0, height = 0, maxVal = 0;
        std::vector<unsigned char> rgb;

        // LOAD immagine
        std::cout << "\nReading PPM from: " << inputPath << std::endl;
        if (!loadPPM(inputPath, width, height, maxVal, rgb)) {
            std::cerr << "Skipping: " << inputPath << "\n";
            continue;
        }

        // RGB -> grayscale
        std::vector<unsigned char> gray(width * height);
        for (int i = 0; i < width * height; i++) {
            int r = rgb[3 * i + 0];
            int g = rgb[3 * i + 1];
            int b = rgb[3 * i + 2];
            gray[i] = static_cast<unsigned char>((r + g + b) / 3);
        }

        size_t bytes = size_t(width) * size_t(height) * sizeof(unsigned char);

        // Reallocate GPU buffers se serve
        if (bytes != allocatedBytes) {
            if (d_in)  cudaFree(d_in);
            if (d_out) cudaFree(d_out);

            cudaError_t err = cudaMalloc(&d_in, bytes);
            if (err != cudaSuccess) {
                std::cerr << "cudaMalloc d_in failed: " << cudaGetErrorString(err) << "\n";
                return 1;
            }

            err = cudaMalloc(&d_out, bytes);
            if (err != cudaSuccess) {
                std::cerr << "cudaMalloc d_out failed: " << cudaGetErrorString(err) << "\n";
                cudaFree(d_in);
                return 1;
            }

            allocatedBytes = bytes;
        }

        cudaMemcpy(d_in, gray.data(), bytes, cudaMemcpyHostToDevice);

        dim3 block(TILE_W, TILE_H);
        dim3 grid((width  + TILE_W - 1) / TILE_W,
                  (height + TILE_H - 1) / TILE_H);

        // Time kernel
        cudaEventRecord(start);
        sobelSharedKernel<<<grid, block>>>(d_in, d_out, width, height);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << "\n";
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);

        // totalKernelMs += ms;
        totalSobelKernelMs += ms;
        imageCount++;

        // std::cout << "CUDA shared kernel time (" << name << "): " << ms << " ms\n";
        std::cout << "SOBEL SHARED - image: " << name
                << " size: " << width << "x" << height
                << " kernel_time_ms: " << ms
                << " grid: " << grid.x << " " << grid.y
                << " block: " << block.x << " " << block.y
                << "\n";



        // D2H
        std::vector<unsigned char> outGray(width * height);
        cudaMemcpy(outGray.data(), d_out, bytes, cudaMemcpyDeviceToHost);

        // Gray -> RGB (B/W) for saving
        std::vector<unsigned char> outRGB(width * height * 3);
        for (int i = 0; i < width * height; i++) {
            unsigned char v = outGray[i];
            outRGB[3 * i + 0] = v;
            outRGB[3 * i + 1] = v;
            outRGB[3 * i + 2] = v;
        }

        // std::cout << "Saving CUDA shared output to: " << outputPath << std::endl;
        std::cout << "Saved Sobel output: " << outputPath << "\n";

        if (!savePPM(outputPath, width, height, maxVal, outRGB)) {
            std::cerr << "Save failed: " << outputPath << "\n";
        }

        // BLUE FILTER

        size_t rgbBytes = size_t(width) * size_t(height) * 3 * sizeof(unsigned char);

        if (rgbBytes != allocatedRgbBytes) {
            if (d_rgb_in)  cudaFree(d_rgb_in);
            if (d_rgb_out) cudaFree(d_rgb_out);

            cudaError_t errB = cudaMalloc(&d_rgb_in, rgbBytes);
            if (errB != cudaSuccess) {
                std::cerr << "cudaMalloc d_rgb_in failed: " << cudaGetErrorString(errB) << "\n";
                return 1;
            }

            errB = cudaMalloc(&d_rgb_out, rgbBytes);
            if (errB != cudaSuccess) {
                std::cerr << "cudaMalloc d_rgb_out failed: " << cudaGetErrorString(errB) << "\n";
                return 1;
            }

            allocatedRgbBytes = rgbBytes;
        }

        // H2D
        cudaMemcpy(d_rgb_in, rgb.data(), rgbBytes, cudaMemcpyHostToDevice);

        // Launch
        dim3 blockB(16, 16);
        dim3 gridB((width  + blockB.x - 1) / blockB.x,
                   (height + blockB.y - 1) / blockB.y);

        cudaEventRecord(start);
        blueKernel<<<gridB, blockB>>>(d_rgb_in, d_rgb_out, width, height);
        cudaError_t errBlue = cudaGetLastError();
        if (errBlue != cudaSuccess) {
            std::cerr << "Blue kernel launch failed: " << cudaGetErrorString(errBlue) << "\n";
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float blueMs = 0.0f;
       

        cudaEventElapsedTime(&blueMs, start, stop);
        totalBlueKernelMs += blueMs;
        // std::cout << "CUDA blue kernel time (" << name << "): " << blueMs << " ms\n";
        std::cout << "BLUE FILTER - image: " << name
                << " size: " << width << "x" << height
                << " kernel_time_ms: " << blueMs
                << " grid: " << gridB.x << " " << gridB.y
                << " block: " << blockB.x << " " << blockB.y
                << "\n";


        // D2H
        std::vector<unsigned char> outBlue(width * height * 3);
        cudaMemcpy(outBlue.data(), d_rgb_out, rgbBytes, cudaMemcpyDeviceToHost);

        // Salvataggio
        std::cout << "Saving CUDA blue filter output to: " << bluePath << std::endl;
        if (!savePPM(bluePath, width, height, maxVal, outBlue)) {
            std::cerr << "Save failed: " << bluePath << "\n";
        }


    }

    auto totalEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> wallMs = totalEnd - totalStart;

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    if (d_in)  cudaFree(d_in);
    if (d_out) cudaFree(d_out);
    if (d_rgb_in)  cudaFree(d_rgb_in);
    if (d_rgb_out) cudaFree(d_rgb_out);

    std::cout << "\nOPTIMIZATION \n";
    std::cout << "Implementation : CUDA Sobel with shared memory + Blue filter\n";
    std::cout << "Dataset: " << inputDir << "\n";
    std::cout << "Output: " << outputDir << "\n";
    std::cout << "Images: " << imageCount << "\n";

    std::cout << "\nTIMINGS \n";
    std::cout << "Total Sobel kernel time: " << totalSobelKernelMs << " ms\n";
    std::cout << "Avg kernel per image: "
            << (imageCount ? totalSobelKernelMs / imageCount : 0.0) << " ms/img\n";

    std::cout << "Total Blue kernel time: " << totalBlueKernelMs << " ms\n";
    std::cout << "Avg Blue kernel per image: "
            << (imageCount ? totalBlueKernelMs / imageCount : 0.0) << " ms/img\n";
    std::cout << "Total wall-clock time: " << wallMs.count() << " ms\n";
    std::cout << "Avg wall-clock per image: "
            << (imageCount ? wallMs.count() / imageCount : 0.0) << " ms/img\n";


    return 0;
}
