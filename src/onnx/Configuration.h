#pragma once

#include <thread>
#include <iostream>

struct Configuration
{   
    int32_t num_thread = std::min(4 , (int32_t)std::thread::hardware_concurrency());

    float confThreshold = 0.50f;
    float iouThreshold = 0.45f;

    bool doVisualize = false;
    bool cudaEnable = false;
    
    std::string ModelPath = "models/yolov8-detect.onnx";
    std::string SavePath = "output";
};
