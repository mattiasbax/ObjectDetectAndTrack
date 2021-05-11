#pragma once
#include <memory>
#include <opencv2/tracking/tracking.hpp>
#include "ObjectTrackerFactory.h"
#include "CommonTypes.h"

class ObjectTrackerHandler {
public:
    struct Parameters
    {
        unsigned int MaxNumberOfTrackedObjects;
        std::vector<int> ClassesToTrack;
    };
    explicit ObjectTrackerHandler(const ObjectTrackerFactory::TrackerType trackerType, Parameters&& parameters = {1,{0}}) : Otf(trackerType), Param(parameters) {};
    [[nodiscard]] std::vector<ObjectDetection> trackObjects(const cv::Mat& image, const std::vector<ObjectDetection>& objectDetections);
private:
    struct TrackedObject
    {
        cv::Ptr<cv::Tracker> Tracker;
        ObjectDetection Object;
    };

    const ObjectTrackerFactory Otf;
    std::vector<TrackedObject> TrackedObjects;
    const Parameters Param;

    bool isObjectAlreadyTracked(const ObjectDetection& object) const;
    void createNewTracker(const cv::Mat& image, const ObjectDetection& object);
};