#pragma once
#include <memory>
#include <opencv2/tracking/tracking.hpp>
#include "ObjectTrackerFactory.h"
#include "CommonTypes.h"

class ObjectTrackerHandler {
public:
    struct Parameters
    {
        std::vector<int> ClassesToTrack;
    };
    explicit ObjectTrackerHandler(const ObjectTrackerFactory::TrackerType trackerType, Parameters&& parameters = {{0}}) : Otf(trackerType), Param(parameters) {};
    [[nodiscard]] std::vector<ObjectDetection> trackObjects(const std::vector<ObjectDetection>& objectDetections);
private:
    const ObjectTrackerFactory Otf;
    const std::vector<cv::Ptr<cv::Tracker>> Tracker;
    const Parameters Param;
};