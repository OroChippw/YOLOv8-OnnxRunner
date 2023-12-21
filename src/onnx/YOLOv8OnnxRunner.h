#pragma once

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

#include "Configuration.h"

typedef struct _DL_RESULT
{
    int classId;
    float confidence;
    cv::Rect box;
    // std::vector<cv::Point2f> keyPoints;
} DETECT_RESULT;

class YOLOv8OnnxRunner
{
private:
    bool cudaEnable;
    int input_width = 640;
    int input_height = 640;
    int num_classes = 1;
    float resizeScales = 1.0f; // Letterbox Scales
    const int reg_max = 16;
    float confThreshold = 0.60f;
    float iouThreshold = 0.45f;
    std::vector<std::string> classes = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", 
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", 
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", 
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", 
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", 
        "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", 
        "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", 
        "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    };

    Ort::Env env;
	Ort::SessionOptions session_options;
    Ort::Session* session = nullptr;
    Ort::RunOptions options;
    Ort::MemoryInfo memory_info_handler = Ort::MemoryInfo::CreateCpu(
		OrtArenaAllocator, OrtMemTypeDefault
	);
    std::vector<float> input_image;
    std::vector<const char*> inputNodeNames;
    std::vector<const char*> outputNodeNames;
    // float32[1,3,640,640]
    std::vector<std::vector<int64_t>> inputNodeDims; 
    // float32[concatoutput0_dim_0,concatoutput_dim_1 , concat_output0_dim_2]
    std::vector<std::vector<int64_t>> outputNodeDims; 

private:
    inline void Softmax();
    inline void Normalize(cv::Mat image);
    void NonMaximumSuppression();

protected:
    void Preprocess(cv::Mat srcImage , cv::Mat& processImage , float* pad_left , float* pad_top);

    void Inference(float*& predict);

    void Postprocess(float* output , std::vector<DETECT_RESULT>& result , float* pad_left , float* pad_top);

public:
    explicit YOLOv8OnnxRunner(Configuration cfg); 
    ~YOLOv8OnnxRunner();

    void InitOrtEnv(Configuration cfg);

    std::vector<DETECT_RESULT> InferenceSingleImage(const cv::Mat& srcImage);

    cv::Mat VisualizationPredicition(cv::Mat image , std::vector<DETECT_RESULT> result);

    void setConfThreshold(float threshold);

    void setNMSThreshold(float threshold);

};