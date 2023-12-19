#pragma once

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

class YOLOv8OnnxRunner
{
private:
    float confThreshold;
    float nmsThreshold;

    const int num_classes = 1;
    const int reg_max = 16;
    const int input_width = 640;
    const int input_height = 640;
    float resizeScales; //letterbox scale

    Ort::Env env;
	Ort::SessionOptions session_options;
    Ort::Session* session;
    bool cudaEnable;
    Ort::RunOptions options;
    std::vector<const char*> inputNodeNames;
    std::vector<const char*> outputNodeNames;

private:
    inline void softmax();

protected:
    cv::Mat Preprocess(cv::Mat srcImage , int *new_h , int *new_w , int *pad_w , int *pad_h);

    cv::Mat Postprocess();

public:
    explicit YOLOv8OnnxRunner(Configuration cfg); 
    ~YOLOv8OnnxRunner();

    void InitOrtEnv(Configuration cfg);

    cv::Mat InferenceSingleImage(const cv::Mat& srcImage);

    void VisualizationPredicition();

    void setConfThreshold(float threshold);

    void setNMSThreshold(float threshold);


};