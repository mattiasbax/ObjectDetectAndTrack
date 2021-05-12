#pragma once

#include <opencv2/core.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <utility>
#include "CommonTypes.h"

class ObjectDetectorHandler
{
public:

    struct Parameters
    {
        float ConfThreshold;
        float NmsThreshold;
        int preferredBackend;
        int preferredTarget;
        float ScaleFactor;
        bool SwapRGB;
        cv::Size InputSize;
        cv::Scalar Mean;
    };

    explicit ObjectDetectorHandler(std::string&& yoloName, Parameters&& parameters = {0.75, 0.4, cv::dnn::DNN_BACKEND_CUDA,
                                                                cv::dnn::DNN_TARGET_CUDA,
                                                                1./255., true, {412, 412}, {0.,0.,0.}}) : YoloName(std::move(yoloName)),
                                                    Param(std::move(parameters)),
                                           Initialized(false), ClassNames{}, OutLayerNames{} {}
    bool init();

    [[nodiscard]] const std::vector<std::string>& getClassNames() const;
    [[nodiscard]] std::vector<Object> detectObjects(const cv::Mat& image);

private:
    const Parameters Param;
    const std::string YoloName;
    bool Initialized;
    cv::dnn::Net Yolo;
    std::vector<std::string> ClassNames;
    std::vector<std::string> OutLayerNames;
};