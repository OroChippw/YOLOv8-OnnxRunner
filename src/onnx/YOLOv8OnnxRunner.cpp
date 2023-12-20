#pragma once

#include <chrono>

#include "Configuration.h"
#include "YOLOv8OnnxRunner.h"

YOLOv8OnnxRunner::YOLOv8OnnxRunner(Configuration cfg)
{
    try
    {
        this->confThreshold = cfg.confThreshold;
        this->nmsThreshold = cfg.nmsThreshold;
        InitOrtEnv(cfg);
    }
    catch(const std::exception& e)
    {
        std::cerr << "[ERROR] : " << e.what() << '\n';
    }
}

YOLOv8OnnxRunner::~YOLOv8OnnxRunner()
{
    delete session;
}

void YOLOv8OnnxRunner::setConfThreshold(float threshold)
{
    this->confThreshold = threshold;
}

void YOLOv8OnnxRunner::setNMSThreshold(float threshold)
{
    this->nmsThreshold = threshold;
}

void YOLOv8OnnxRunner::Softmax()
{

}

void YOLOv8OnnxRunner::Normalize(cv::Mat image)
{
    int imageWidth = image.cols;
    int imageHeight = image.rows;
    int imageChannel = image.channels();
    this->input_image.resize(imageWidth * imageHeight * imageChannel);

    for (int c = 0 ; c < imageChannel ; c++)
    {
        for (int h = 0 ; h < imageHeight ; h++)
        {
            for (int w = 0 ; w < imageWidth ; w++)
            {
                this->input_image[c * imageWidth * imageHeight + h * imageWidth + w] = \
                    image.at<cv::Vec3b>(h , w)[c] / 255.0f;
            }
        }
    }
}

void YOLOv8OnnxRunner::NonMaximumSuppression()
{
    // sort();
}

void YOLOv8OnnxRunner::InitOrtEnv(Configuration cfg)
{
    env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "YOLOv8Model");
    if (cfg.cudaEnable)
    {
        OrtCUDAProviderOptions cudaOption;
        cudaOption.device_id = 0;
        session_options.AppendExecutionProvider_CUDA(cudaOption);
    }

	session_options = Ort::SessionOptions();
	session_options.SetInterOpNumThreads(cfg.num_thread);
	session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    std::wstring model_path = std::wstring(cfg.ModelPath.begin() , cfg.ModelPath.end());
    session = new Ort::Session(env , model_path.c_str() , session_options);

    Ort::AllocatorWithDefaultOptions allocator;
    size_t inputNodesNum = session->GetInputCount();
    for (size_t i = 0; i < inputNodesNum; i++)
    {
        inputNodeNames.push_back(session->GetInputNameAllocated(i , allocator).get());
        inputNodeDims.push_back(session->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
    }
    
    size_t OutputNodesNum = session->GetOutputCount();
    for (size_t i = 0; i < OutputNodesNum; i++)
    {
        outputNodeNames.push_back(session->GetOutputNameAllocated(i , allocator).get());
        outputNodeDims.push_back(session->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
    }

    this->input_width = inputNodeDims[0][2];
    this->input_height = inputNodeDims[0][3];

    std::cout << "[INFO] Build Session successfully." << std::endl;
}

void YOLOv8OnnxRunner::Preprocess(cv::Mat srcImage , cv::Mat& processImage)
{   
    int src_width = srcImage.cols , src_height = srcImage.rows;

    if (srcImage.channels() == 3)
    {
        processImage = srcImage.clone();
        cv::cvtColor(processImage , processImage , cv::COLOR_BGR2RGB);
    } else 
    {
        cv::cvtColor(srcImage , processImage , cv::COLOR_GRAY2RGB);
    }

    if (src_width >= src_height)
    {
        resizeScales = src_width / (float)this->input_width;
        cv::resize(processImage, processImage, cv::Size(this->input_width, int(src_height / resizeScales)));
    } else
    {
        resizeScales = src_height / (float)this->input_height;
        cv::resize(processImage , processImage , cv::Size(int(src_width / resizeScales) , this->input_height));
    }

    cv::Mat tempImage = cv::Mat::zeros(this->input_width , this->input_height , CV_8UC3);
    processImage.copyTo(tempImage(cv::Rect(0 , 0 , processImage.cols , processImage.rows)));
    processImage = tempImage;

    Normalize(processImage);
}

void YOLOv8OnnxRunner::Inference(float* result)
{   
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>( \
        memory_info_handler , this->input_image.data() , this->input_image.size() , inputNodeDims[0].data() , inputNodeDims[0].size());
    
    std::cout << "[INFO] Inference Start ..." << std::endl;
    
    auto time_start = std::chrono::high_resolution_clock::now();
    auto output_tensor = session->Run(
        options , inputNodeNames.data() , &input_tensor , 1 , outputNodeNames.data() , outputNodeNames.size());
    auto time_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = time_end - time_start;

    std::cout << "[INFO] Inference Finish ..." << std::endl;
    std::cout << "[INFO] Inference Cost time : " << diff.count() << "s" << std::endl;

    result = output_tensor.front().GetTensorMutableData<float>();
}

void YOLOv8OnnxRunner::Postprocess(float* output , std::vector<DETECT_RESULT>& result)
{
    int strideNum = outputNodeDims[0][1]; // 8400
    int signalResultNum = outputNodeDims[0][2]; // 84
    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    cv::Mat rawData;

    rawData = cv::Mat(strideNum , signalResultNum , CV_32F , output);
    float* data = (float*)rawData.data;

    for (int i = 0 ; i < strideNum ; ++i)
    {
        float* classesScores = data + 4;
        cv::Mat scores(1 , this->classes.size() , CV_32FC1, classesScores);
        cv::Point class_id;
        double maxClassScore;
        cv::minMaxLoc(scores , 0 , &maxClassScore , 0 , &class_id);
        if (maxClassScore > this->confThreshold)
        {
            confidences.emplace_back(maxClassScore);
            class_ids.emplace_back(class_id.x);
            float x = data[0];
            float y = data[1];
            float w = data[2];
            float h = data[3];

            int left = int((x - 0.5 * w) * this->resizeScales);
            int top = int((y - 0.5 * h) * this->resizeScales);

            int width = int(w * this->resizeScales);
            int height = int(h * this->resizeScales);

            boxes.emplace_back(cv::Rect(left , top , width , height));
        }
        data += signalResultNum;
    }

    std::vector<int> nmsResult;
    cv::dnn::NMSBoxes(boxes , confidences , this->confThreshold , this->nmsThreshold , nmsResult);

    for (int i = 0 ; i < nmsResult.size() ; i++)
    {
        int idx = nmsResult[i];
        DETECT_RESULT res;
        res.classId = class_ids[idx];
        res.confidence = confidences[idx];
        res.box = boxes[idx];
        result.emplace_back(res);
    }
}

cv::Mat YOLOv8OnnxRunner::VisualizationPredicition(cv::Mat image , std::vector<DETECT_RESULT> result)
{
    for (auto& re : result)
    {
        cv::RNG rng(cv::getTickCount());
        cv::Scalar color(rng.uniform(0 , 256) , rng.uniform(0 , 256) , rng.uniform(0 , 256));
        cv::rectangle(image , re.box , color , 3);
        
        float confidence = float(100 * re.confidence) / 100;
        std::cout << std::fixed << std::setprecision(2);
        std::string label = this->classes[re.classId] + " " + \
            std::to_string(confidence).substr(0 , std::to_string(confidence).size() - 4);
        
        cv::rectangle(image , cv::Point(re.box.x , re.box.y - 25) , \
            cv::Point(re.box.x + label.length() * 15 , re.box.y) , color , cv::FILLED);
        
        cv::putText(image , label , cv::Point(re.box.x , re.box.y - 5) , \
            cv::FONT_HERSHEY_SIMPLEX , 0.75 , cv::Scalar(0,0,0) , 2);
    }

    return image;
}

std::vector<DETECT_RESULT> YOLOv8OnnxRunner::InferenceSingleImage(const cv::Mat& srcImage)
{
    float* predict;
    cv::Mat processImage;
    std::vector<DETECT_RESULT> result;
    int new_w = 0 , new_h = 0 , pad_w = 0 , pad_h = 0;

    Preprocess(srcImage , processImage);
    
    Inference(predict);
    
    Postprocess(predict , result);
    
    return result;
}

