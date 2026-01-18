#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <filesystem>
#include <chrono>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace fs = std::filesystem;

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
        std::cerr << "Unsupported PPM type (expected P3): " << type << std::endl;
        return false;
    }

    // Salta eventuali commenti
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
// RIF: M3_CUDA_Parallel_Model.pdf 
// Kernel CUDA (__global__): eseguito su device e lanciato da host
// Data-parallelism: ogni thread calcola 1 elemento di output (1 pixel)
// Mapping indici: i = threadIdx.x + blockIdx.x * blockDim.x
__global__ void sobelKernel(const unsigned char* inputGray,
                            unsigned char* outputGray,
                            int width, int height)
{
    // RIF: M3_CUDA_Parallel_Model.pdf p.11 e p.26
    // Immagini -> griglia 2D: x,y derivano da (blockIdx, threadIdx).
    // Linearizzazione row-major: idx = y * width + x.
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // RIF: M5_Thread_Execution_Efficiency.pdf p.8
    // control divergence
    if (x <= 0 || x >= width - 1 || y <= 0 || y >= height - 1)
        return;

    int idx = y * width + x;

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

    for (int ky = -1; ky <= 1; ky++) {
        for (int kx = -1; kx <= 1; kx++) {
            int pidx = (y + ky) * width + (x + kx);
            int pixel = inputGray[pidx];
            sumX += pixel * Gx[ky + 1][kx + 1];
            sumY += pixel * Gy[ky + 1][kx + 1];
        }
    }

    float mag = sqrtf(float(sumX * sumX + sumY * sumY));
    outputGray[idx] = (mag >= 100.0f) ? 255 : 0;
}

//FILTRO BLU
// RIF: M3_CUDA_Parallel_Model.pdf p.4 (1 thread -> 1 elemento) + M6_Memory_Access_Performance.pdf p.12
// Blue filter = embarrassingly parallel: ogni thread aggiorna 1 pixel RGB indipendente
__global__ void blueKernel(const unsigned char* inRGB,
                           unsigned char* outRGB,
                           int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    // RIF: M6_Memory_Access_Performance.pdf p.14
    // Accesso coalesced 
    //Coalescing: se gli accessi del warp cadono nella stessa burst section --> 1 richiesta DRAM --> altrimenti più richieste

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


int main()
{
    // fs::create_directories("../data/output/logs/");
    // // SAVE logs into txt file
    // std::ofstream logFile("../data/output/logs/cuda_naive_output.txt");
    // std::streambuf* originalCout = std::cout.rdbuf();

    // if (logFile.is_open()) {
    //     std::cout.rdbuf(logFile.rdbuf());
    // }
    std::string inputDir  = "../data/input/dataset/ppm/";

    std::string outputDir = "../data/output/sobel_naive/";
    std::string sobelDir   = outputDir + "sobel/";
    std::string blueDir    = outputDir + "blue/";
    fs::create_directories(sobelDir);
    fs::create_directories(blueDir);

    // std::cout << "CUDA Sobel naive benchmark on dataset: " << inputDir << "\n";
    std::cout << "CUDA BENCHMARK NAIVE\n";
    std::cout << "Dataset: " << inputDir << "\n";
    std::cout << "Output folder: " << outputDir << "\n";

    // Reuse GPU buffers if size stays the same
    unsigned char *d_in = nullptr, *d_out = nullptr;
    size_t allocatedBytes = 0;
    unsigned char *d_rgb_in = nullptr, *d_rgb_out = nullptr;
    size_t allocatedRgbBytes = 0;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int imageCount = 0;
    // double totalKernelMs = 0.0;
    double totalSobelKernelMs = 0.0;
    double totalBlueKernelMs  = 0.0;


    auto totalStart = std::chrono::high_resolution_clock::now();

    for (const auto& entry : fs::directory_iterator(inputDir))
    {
        if (entry.path().extension() != ".ppm")
            continue;

        std::string inputPath  = entry.path().string();
        std::string name       = entry.path().stem().string();
        std::string outputPath = sobelDir + name + "_sobel_cuda.ppm";
        std::string bluePath  = blueDir + name + "_blue_cuda.ppm";

        int width = 0, height = 0, maxVal = 0;
        std::vector<unsigned char> rgb;

        std::cout << "\nReading PPM from: " << inputPath << std::endl;
        if (!loadPPM(inputPath, width, height, maxVal, rgb)) {
            std::cerr << "Skipping (load failed): " << inputPath << "\n";
            continue;
        }

        // grayscale CPU
        std::vector<unsigned char> gray(width * height);
        for (int i = 0; i < width * height; i++) {
            int r = rgb[3 * i + 0];
            int g = rgb[3 * i + 1];
            int b = rgb[3 * i + 2];
            gray[i] = static_cast<unsigned char>((r + g + b) / 3);
        }

        size_t bytes = size_t(width) * size_t(height) * sizeof(unsigned char);

        // Riallocazion se dimensioni cambiano
        if (bytes != allocatedBytes) {
            if (d_in)  cudaFree(d_in);
            if (d_out) cudaFree(d_out);
        
        // RIF: M2_Intro_to_CUDA_C.pdf p.17
        // Allocazione memoria su device (cudaMalloc)

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

        // RIF: M2_Intro_to_CUDA_C.pdf p.18
        // cudaMemcpy(): trasferimento Host - Device.

        cudaMemcpy(d_in, gray.data(), bytes, cudaMemcpyHostToDevice);

        // RIF: M3_CUDA_Parallel_Model.pdf p.3
        // Concetto ceil -> uso (D + B - 1)/B per coprire tutti i pixel anche se N non è multiplo di B
        dim3 block(16, 16);
        dim3 grid((width  + block.x - 1) / block.x,
                  (height + block.y - 1) / block.y);

        // time kernel
        cudaEventRecord(start);
        sobelKernel<<<grid, block>>>(d_in, d_out, width, height);
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

        // std::cout << "CUDA kernel time (" << name << "): " << ms << " ms\n";
        std::cout << "SOBEL NAIVE - image: " << name
                << " size: " << width << "x" << height
                << " kernel_time_ms: " << ms
                << " grid: " << grid.x << " " << grid.y
                << " block: " << block.x << " " << block.y
                << "\n";

        // D2H
        std::vector<unsigned char> outGray(width * height);
        cudaMemcpy(outGray.data(), d_out, bytes, cudaMemcpyDeviceToHost);

        // gray -> RGB 
        std::vector<unsigned char> outRGB(width * height * 3);
        for (int i = 0; i < width * height; i++) {
            unsigned char v = outGray[i];
            outRGB[3 * i + 0] = v;
            outRGB[3 * i + 1] = v;
            outRGB[3 * i + 2] = v;
        }

        std::cout << "Saving CUDA output to: " << outputPath << std::endl;
        if (!savePPM(outputPath, width, height, maxVal, outRGB)) {
            std::cerr << "Save failed: " << outputPath << "\n";
        }

        // FITLRO BLU
        
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

        // launch
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

        // std::cout << "CUDA blue kernel time (" << name << "): " << blueMs << " ms\n";
        std::cout << "BLUE FILTER - image: " << name
                << " size: " << width << "x" << height
                << " kernel_time_ms: " << blueMs
                << " grid: " << gridB.x << " " << gridB.y
                << " block: " << blockB.x << " " << blockB.y
                << "\n";
        
        // totalKernelMs += blueMs;
        totalBlueKernelMs += blueMs;



        // D2H + save
        std::vector<unsigned char> outBlue(width * height * 3);
        cudaMemcpy(outBlue.data(), d_rgb_out, rgbBytes, cudaMemcpyDeviceToHost);

        std::cout << "Saving CUDA blue output to: " << bluePath << std::endl;
        if (!savePPM(bluePath, width, height, maxVal, outBlue)) {
            std::cerr << "Save failed: " << bluePath << "\n";
        }

    }

    auto totalEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> wallMs = totalEnd - totalStart;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    if (d_in)  cudaFree(d_in);
    if (d_out) cudaFree(d_out);
    if (d_rgb_in)  cudaFree(d_rgb_in);
    if (d_rgb_out) cudaFree(d_rgb_out);

    std::cout << "\nOPTIMIZATION \n";
    std::cout << "Implementation: CUDA naive Sobel + Blue filter\n";
    std::cout << "Dataset: " << inputDir << "\n";
    std::cout << "Output: " << outputDir << "\n";
    std::cout << "Images processed: " << imageCount << "\n";

    std::cout << "\nTimings\n";
    std::cout << "Total sobel kernel time ms: " << totalSobelKernelMs << "\n";
    std::cout << "Average Sobel kernel time ms per image: "
            << (imageCount ? totalSobelKernelMs / imageCount : 0.0) << "\n";
    std::cout << "Total Blue kernel time ms: " << totalBlueKernelMs << "\n";
    std::cout << "Average Blue kernel time ms per image: "
            << (imageCount ? totalBlueKernelMs / imageCount : 0.0) << "\n";
    std::cout << "Total wall clock time ms: " << wallMs.count() << "\n";
    std::cout << "Average wall clock time ms per image: "
            << (imageCount ? wallMs.count() / imageCount : 0.0) << "\n";

    return 0;
}
