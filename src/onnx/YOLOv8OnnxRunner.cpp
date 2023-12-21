#pragma once

#include <chrono>

#include "Configuration.h"
#include "YOLOv8OnnxRunner.h"

YOLOv8OnnxRunner::YOLOv8OnnxRunner(Configuration cfg)
{
    try
    {
        this->confThreshold = cfg.confThreshold;
        this->iouThreshold = cfg.iouThreshold;
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
    this->iouThreshold = threshold;
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
        char* temp_buf = new char[50];
        strcpy(temp_buf , session->GetInputNameAllocated(i , allocator).get());
        inputNodeNames.push_back(temp_buf);
        inputNodeDims.push_back(session->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
    }
    std::cout << "[INFO] InputNodeNum : " << inputNodesNum << " InputNodeName : ";
    for (int idx = 0 ; idx < inputNodeNames.size() ; idx++)
    {
        std::cout << inputNodeNames[idx] << " ";
    }
    std::cout << std::endl;

    size_t OutputNodesNum = session->GetOutputCount();
    for (size_t i = 0; i < OutputNodesNum; i++)
    {   
        char* temp_buf = new char[50];
        strcpy(temp_buf , session->GetOutputNameAllocated(i , allocator).get());
        outputNodeNames.push_back(temp_buf);
        outputNodeDims.push_back(session->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
    }
    std::cout << "[INFO] OutputNodesNum : " << OutputNodesNum << " OutputNodeName : ";
    for (int idx = 0 ; idx < outputNodeNames.size() ; idx++)
    {
        std::cout << outputNodeNames[idx] << " ";
    }
    std::cout << std::endl;

    this->input_width = inputNodeDims[0][2];
    this->input_height = inputNodeDims[0][3];

    std::cout << "[INFO] Build Session successfully." << std::endl;
}


void YOLOv8OnnxRunner::Preprocess(cv::Mat srcImage , cv::Mat& processImage , float* pad_left , float* pad_top)
{   
    std::cout << "[INFO] PreProcess Image ..." << std::endl;
    int src_width = srcImage.cols , src_height = srcImage.rows;
    std::cout << "[INFO] Image width : " << src_width << " , hegiht : " << src_height << std::endl;
    if (srcImage.channels() == 3)
    {
        processImage = srcImage.clone();
        // cv::cvtColor(processImage , processImage , cv::COLOR_BGR2RGB);
    } else 
    {
        cv::cvtColor(srcImage , processImage , cv::COLOR_GRAY2RGB);
    }

    this->resizeScales = std::min((float)this->input_width / (float)src_width , 
        (float)this->input_height / (float)src_height);
    std::cout << "[INFO] Set resizeScales : " << this->resizeScales << std::endl;
    int new_un_pad[2] = { (int)std::round((float)src_width * this->resizeScales) , \
                            (int)std::round((float)src_height * this->resizeScales) };
    // std::cout << new_un_pad[0] << " " << new_un_pad[1] << std::endl;

    auto dw = (float)(this->input_width - new_un_pad[0]);
	auto dh = (float)(this->input_height - new_un_pad[1]);
    dw /= 2.0f;
	dh /= 2.0f;
    std::cout << "[INFO] dw : " << dw << " dh : " << dh << std::endl;

    if (src_width != new_un_pad[0] && src_height != new_un_pad[1])
	{
		cv::resize(processImage, processImage, cv::Size(new_un_pad[0], new_un_pad[1]));
        std::cout << "[INFO] resizeImage width : " << processImage.cols << ", resizeImage height : " << processImage.rows << std::endl;
	}

    int top = int(std::round(dh - 0.1f));
	int bottom = int(std::round(dh + 0.1f));
	int left = int(std::round(dw - 0.1f));
	int right = int(std::round(dw + 0.1f));

    *pad_left = left;
    *pad_top = top;

	cv::copyMakeBorder(processImage, processImage, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(114,114,144));
    // cv::imshow("processImage" , processImage);
    Normalize(processImage);

    std::cout << "[INFO] processImage width : " << processImage.cols << ", processImage height : " << processImage.rows << std::endl;
}

void YOLOv8OnnxRunner::Inference(float*& result)
{   
    try
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

        auto temp = output_tensor[0].GetTensorMutableRawData();
        auto temp_dims = output_tensor[0].GetTypeInfo().GetTensorTypeAndShapeInfo().GetShape();

        std::cout << "[INFO] Concatoutput0_dim_0 : "<< static_cast<int>(temp_dims.at(0)) \
                    << ", Concatoutput0_dim_1 : " << static_cast<int>(temp_dims.at(1)) \
                    << ", Concatoutput0_dim_2 : " << static_cast<int>(temp_dims.at(2)) << std::endl;

        result = output_tensor[0].GetTensorMutableData<float>();
    }
    catch(const std::exception& e)
    {
        std::cerr << "[ERROR] : " << e.what() << '\n';
    }
}

void YOLOv8OnnxRunner::Postprocess(float* output , std::vector<DETECT_RESULT>& result , float* pad_left , float* pad_top)
{
    std::cout << "[INFO] Postprocess Start ..." << std::endl;
    int strideNum = outputNodeDims[0][1]; // 84
    int signalResultNum = outputNodeDims[0][2]; // 8400
    int score_array_length = (int)outputNodeDims[0][1] - 4;
    std::cout << "[INFO] strideNum : " << strideNum << " , signalResultNum : " << signalResultNum << std::endl;

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    cv::Mat rawData = cv::Mat(cv::Size((int)outputNodeDims[0][2] , (int)outputNodeDims[0][1])  , CV_32F , output).t();
    float* data = (float*)rawData.data;
    std::cout << "[INFO] rawData width : " << rawData.cols << ", rawData height : " << rawData.rows << std::endl;
    
    for (int i = 0 ; i < signalResultNum ; ++i)
    {
        float* classesScores = data + 4;
        cv::Mat scores(1 , score_array_length , CV_32F, data + 4);
        cv::Point class_id;
        double maxClassScore;
        cv::minMaxLoc(scores , 0 , &maxClassScore , 0 , &class_id);
        maxClassScore = (float)maxClassScore;
        if (maxClassScore >= this->confThreshold)
        {
            confidences.emplace_back(maxClassScore);
            class_ids.emplace_back(class_id.x);
            // [x,y,w,h]
            float x = (data[0] - *pad_left) / this->resizeScales;
            float y = (data[1] - *pad_top) / this->resizeScales;
            float w = data[2] / this->resizeScales;
            float h = data[3] / this->resizeScales;

            int left = std::max(int(x - 0.5 * w + 0.5), 0);
            int top = std::max(int(y - 0.5 * h + 0.5), 0);

            boxes.emplace_back(cv::Rect(left , top , int(w + 0.5), int(h + 0.5)));
        }
        data += strideNum;
    }

    std::vector<int> nmsResult;
    cv::dnn::NMSBoxes(boxes , confidences , this->confThreshold , this->iouThreshold , nmsResult);
    std::cout << "[INFO] NMSResult Size : " << nmsResult.size() << std::endl;
    for (int i = 0 ; i < nmsResult.size() ; i++)
    {
        int idx = nmsResult[i];
        DETECT_RESULT res;
        res.classId = class_ids[idx];
        res.confidence = confidences[idx];
        res.box = boxes[idx];

        std::cout << "[INFO] classId : " << res.classId << " , className : " << this->classes[res.classId] << " , Confidence : " << res.confidence << " , Box : " << res.box << std::endl;

        result.emplace_back(res);
    }
    std::cout << "[INFO] Postprocess Finish ..." << std::endl;
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
    float* predict = nullptr;
    cv::Mat processImage;
    std::vector<DETECT_RESULT> result;
    float pad_left = 0 , pad_top = 0;

    Preprocess(srcImage , processImage , &pad_left , &pad_top);
    
    Inference(predict);
    
    Postprocess(predict , result ,  &pad_left , &pad_top);

    return result;
}

