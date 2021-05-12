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
    [[nodiscard]] std::vector<Object> trackObjects(const cv::Mat& image, const std::vector<Object>& objectDetections);
private:
    struct TrackedObject
    {
        cv::Ptr<cv::Tracker> Tracker;
        Object TrackedObject;
    };

    const ObjectTrackerFactory Otf;
    std::vector<TrackedObject> TrackedObjects;
    const Parameters Param;

    bool isObjectAlreadyTracked(const Object& object) const;
    void createNewTracker(const cv::Mat& image, const Object& detectedObject);
};