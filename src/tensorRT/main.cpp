/*
    # Authore : OroChippw
    # Last Change : 2023.12.19
*/
#include <iostream>
#include <opencv2/opencv.hpp>

#include "Configuration.h"
#include "YOLOv8TensorRTRunner.h"

void Print_Usage(int argc, char ** argv, const Configuration & cfg)
{
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h, --help            show this help message and exit\n");
    fprintf(stderr, "  -m FNAME, --model-path FNAME\n");
    fprintf(stderr, "                        model path (default: %s)\n", cfg.ModelPath.c_str());
    fprintf(stderr, "  -conf T, --conf-threshold T     detection confidence threshold (default: %.2f)\n", cfg.confThreshold);
    fprintf(stderr, "  -nms T, --nms-threshold T     non maximum suppression threshold (default: %.2f)\n", cfg.nmsThreshold);
    fprintf(stderr, "  -img FNAME, --image-path FNAME\n");
    fprintf(stderr, "                        input file \n");
    fprintf(stderr, "  -save FNAME, --save-path FNAME\n");
    fprintf(stderr, "                        output file (default: %s)\n", cfg.SavePath.c_str());
    fprintf(stderr, "\n");
}

bool Params_Parse(int argc , char ** argv , Configuration & cfg , std::string & image_path)
{
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-m" || arg == "--model-path") {
            cfg.ModelPath = argv[++i];
        } else if (arg == "-conf" || arg == "--conf-threshold")
        {
            cfg.confThreshold = std::stof(argv[++i]);
        } else if (arg == "-nms" || arg == "--nms-threshold")
        {
            cfg.nmsThreshold = std::stof(argv[++i]);
        } else if (arg == "-img" || arg == "--image-path")
        {
            image_path = argv[++i];
        } else if (arg == "-save" || arg == "--save-path")
        {
            cfg.SavePath = std::stof(argv[++i]);
        } else if (arg == "-h" || arg == "--help")
        {
            Print_Usage(argc , argv , cfg);
            return EXIT_SUCCESS;
        } else 
        {
            fprintf(stderr , "[ERROR] : Unknown argument : %s\n" , arg.c_str());
            Print_Usage(argc , argv , cfg);
            return EXIT_FAILURE;
        }

    }
    return EXIT_SUCCESS;
}


int main(int argc , char *argv[])
{
    std::string image_path;
    Configuration cfg;

    if (!Params_Parse(argc , argv , cfg , image_path))
    {
        return EXIT_FAILURE;
    }

    YOLOv8TensorRTRunner Detector(cfg);

    cv::Mat srcImage = cv::imread(image_path, -1);

    Detector.InferenceSingleImage(srcImage);
    
    return EXIT_SUCCESS;
}