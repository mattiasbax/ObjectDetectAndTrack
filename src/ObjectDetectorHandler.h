#pragma once

#include <opencv2/core.hpp>
#include <opencv2/dnn/dnn.hpp>
//#include <map>
//#include <utility>


class ObjectDetectorHandler
{
public:
    //enum class SizeType
    //{
    //    Medium,
    //};

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

    struct ObjectDetection
    {
        int classId;
        double confidence;
        cv::Rect boundingBox;
    };

    explicit ObjectDetectorHandler(std::string&& yoloName, Parameters parameters = {0.75, 0.4, cv::dnn::DNN_BACKEND_CUDA,
                                                                cv::dnn::DNN_TARGET_CUDA,
                                                                1./255., true, {412, 412}, {0.,0.,0.}}) : YoloName(std::move(yoloName)),
                                                    Param(parameters),
                                           Initialized(false), ClassNames{}, OutLayerNames{} {}
    bool init();

    [[nodiscard]] const std::vector<std::string>& getClassNames() const;
    [[nodiscard]] std::vector<ObjectDetection> detectObjects(const cv::Mat& image);

private:
    const Parameters Param;
    const std::string YoloName;
    bool Initialized;
    cv::dnn::Net Yolo;
    std::vector<std::string> ClassNames;
    std::vector<std::string> OutLayerNames;

    //const std::map<SizeType, cv::Size> SizeMap;
    //static std::map<SizeType,cv::Size> createMap()
    //{
    //    std::map<SizeType,cv::Size> m;
    //    m[SizeType::Medium] = cv::Size(412,412);
    //    return m;
    //}
};