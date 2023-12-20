#pragma once

#include <iostream>

struct Configuration
{   
    int32_t num_thread = std::min(4 , (int32_t)std::thread::hardware_concurrency());

    float confThreshold = 0.90f;
    float nmsThreshold = 0.50f;

    bool doVisualize = false;
    bool cudaEnable = false;
    
    std::string ModelPath = "models/yolov8-detect.onnx";
    std::string SavePath = "output";
};
