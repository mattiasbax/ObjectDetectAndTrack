#pragma once

#include <opencv2/core.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <utility>

#include "CommonTypes.h"

class ObjectDetectorHandler {
   public:
    struct Parameters {
        Parameters(){};  // NOLINT(modernize-use-equals-default)
        std::vector<int> ClassesToDetect = {2};
        float ConfThreshold = 0.75;
        float NmsThreshold = 0.4;
        int preferredBackend = cv::dnn::DNN_BACKEND_CUDA;
        int preferredTarget = cv::dnn::DNN_TARGET_CUDA;
        float ScaleFactor = 1. / 255.;
        bool SwapRGB = true;
        cv::Size InputSize = {412, 412};
        cv::Scalar Mean = {0., 0., 0.};
    };

    explicit ObjectDetectorHandler(std::string&& yoloName, Parameters&& parameters = Parameters())
        : YoloName(std::move(yoloName)),
          Param(std::move(parameters)),
          Initialized(false),
          ClassNames{},
          OutLayerNames{} {}

    bool init();

    [[nodiscard]] std::vector<Object> detectObjects(const cv::Mat& image);

   private:
    const Parameters Param;
    const std::string YoloName;
    bool Initialized;
    cv::dnn::Net Yolo;
    std::vector<std::string> ClassNames;
    std::vector<std::string> OutLayerNames;
};