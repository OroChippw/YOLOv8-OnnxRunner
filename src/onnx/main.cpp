/*
    # Authore : OroChippw
    # Last Change : 2023.12.20
*/
#include <chrono>
#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>

#include "Configuration.h"
#include "YOLOv8OnnxRunner.h"

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
    fprintf(stderr, "  -v , --visual     visualiztion prediction result (default: %d)\n", cfg.doVisualize);
    fprintf(stderr, "  --cuda     using GPUs for inference (default: %d)\n", cfg.cudaEnable);
    fprintf(stderr, "  -img FNAME, --image-dir FNAME\n");
    fprintf(stderr, "                        input file dir \n");
    fprintf(stderr, "  -save FNAME, --save-path FNAME\n");
    fprintf(stderr, "                        output file (default: %s)\n", cfg.SavePath.c_str());
    fprintf(stderr, "\n");
}

bool Params_Parse(int argc , char ** argv , Configuration & cfg , std::filesystem::path & image_dir)
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
        } else if (arg == "-img" || arg == "--image-dir")
        {
            image_dir = argv[++i];
        } else if (arg == "-save" || arg == "--save-path")
        {
            cfg.SavePath = std::stof(argv[++i]);
        } else if (arg == "-v" || arg == "--visual")
        {
            cfg.doVisualize = true;
        } else if (arg == "--cuda")
        {
            cfg.cudaEnable = true;
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
    std::filesystem::path image_dir;
    Configuration cfg;

    if (!Params_Parse(argc , argv , cfg , image_dir))
    {
        return EXIT_FAILURE;
    }

    YOLOv8OnnxRunner Detector(cfg);

    for (auto& i : std::filesystem::directory_iterator(image_dir))
    {
        if (i.path().extension() == ".jpg" || i.path().extension() == ".png" || i.path().extension() == ".jpeg")
        {
            cv::Mat srcImage = cv::imread(i.path().string(), -1);
            auto result = Detector.InferenceSingleImage(srcImage);
            if (cfg.doVisualize)
            {
                cv::Mat visualImage = Detector.VisualizationPredicition(srcImage , result);
                
                std::cout << "[OPERATION] Press any key to exit" << std::endl;
                cv::imshow("YOLOv8Detect Result" , visualImage);
                cv::waitKey(0);
                cv::destroyAllWindows();
            }
        }
    }

    return EXIT_SUCCESS;
}