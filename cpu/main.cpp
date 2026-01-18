#include <chrono>
#include <iostream>
#include <filesystem>
#include <string>
#include <fstream>
#include "ImageProcessor.h"

namespace fs = std::filesystem;

int main()
{
    // fs::create_directories("../data/output/logs/");
    // // SAVE logs into txt file
    // std::ofstream logFile("../data/output/logs/cpu_output.txt");
    // std::streambuf* originalCout = std::cout.rdbuf();

    // if (logFile.is_open()) {
    //     std::cout.rdbuf(logFile.rdbuf());
    // }
    // std::string inputDir  = "../data/input/dataset_1/";
    std::string inputDir  = "../data/input/dataset/ppm/";

    // Output structure
    std::string outputBase = "../data/output/cpu/";
    std::string sobelDir   = outputBase + "sobel/";
    std::string blueDir    = outputBase + "blue/";

    // Make sure folders exist
    fs::create_directories(sobelDir);
    fs::create_directories(blueDir);

    std::cout << "CPU BENCHMARK\n";
    std::cout << "Implementation: CPU sequential Sobel and Blue filter\n";
    std::cout << "Input dataset: " << inputDir << "\n";
    std::cout << "Output folder: " << outputBase << "\n";

    double totalSobelMs = 0.0;
    double totalBlueMs  = 0.0;
    int imageCount = 0;

    auto totalStart = std::chrono::high_resolution_clock::now();

    for (const auto& entry : fs::directory_iterator(inputDir))
    {
        if (entry.path().extension() != ".ppm")
            continue;

        std::string inputPath  = entry.path().string();
        std::string imageName  = entry.path().stem().string();
        std::string sobelPath  = sobelDir + imageName + "_sobel_cpu.ppm";
        std::string bluePath   = blueDir  + imageName + "_blue_cpu.ppm";

        std::cout << "\nProcessing image: " << imageName << "\n";

        ImageProcessor processor(inputPath);

        // Sobel timing
        auto sobelStart = std::chrono::high_resolution_clock::now();
        processor.sobelFilter(sobelPath);
        auto sobelEnd = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> sobelElapsed = sobelEnd - sobelStart;
        totalSobelMs += sobelElapsed.count();

        std::cout << "SOBEL CPU - image: " << imageName
                  << " time_ms: " << sobelElapsed.count() << "\n";
        std::cout << "Saved Sobel output: " << sobelPath << "\n";

        // Blue timing
        auto blueStart = std::chrono::high_resolution_clock::now();
        processor.blueFilter(bluePath);
        auto blueEnd = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> blueElapsed = blueEnd - blueStart;
        totalBlueMs += blueElapsed.count();

        std::cout << "BLUE CPU - image: " << imageName
                  << " time_ms: " << blueElapsed.count() << "\n";
        std::cout << "Saved Blue output: " << bluePath << "\n";

        imageCount++;
    }

    auto totalEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> wallMs = totalEnd - totalStart;

    std::cout << "\nOPTIMIZATIONS\n";
    std::cout << "Implementation: CPU sequential Sobel and Blue filter\n";
    std::cout << "Dataset: " << inputDir << "\n";
    std::cout << "Output: " << outputBase << "\n";
    std::cout << "Images processed: " << imageCount << "\n";

    std::cout << "\nTIMINGS Sobel\n";
    std::cout << "Total Sobel time ms: " << totalSobelMs << "\n";
    std::cout << "Average Sobel time ms per image: "
              << (imageCount ? totalSobelMs / imageCount : 0.0) << "\n";

    std::cout << "\nTIMINGS Blue\n";
    std::cout << "Total Blue time ms: " << totalBlueMs << "\n";
    std::cout << "Average Blue time ms per image: "
              << (imageCount ? totalBlueMs / imageCount : 0.0) << "\n";

    std::cout << "\nWall clock\n";
    std::cout << "Total wall clock time ms: " << wallMs.count() << "\n";
    std::cout << "Average wall clock time ms per image: "
              << (imageCount ? wallMs.count() / imageCount : 0.0) << "\n";
    // Restore stdout
    // std::cout.rdbuf(originalCout);
    // logFile.close();
    return 0;
}
