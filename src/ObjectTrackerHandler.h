#pragma once
#include <memory>
#include <opencv2/tracking/tracking.hpp>
#include <utility>
#include "ObjectTrackerFactory.h"
#include "CommonTypes.h"

class ObjectTrackerHandler
{
public:
    struct Parameters
    {
        Parameters() {} // NOLINT(modernize-use-equals-default)
        unsigned int MaxNumberOfTrackedObjects = 1;
        std::vector<int> ClassesToTrack = {1};
    };

    explicit ObjectTrackerHandler(const ObjectTrackerFactory::TrackerType trackerType, Parameters&&  parameters = Parameters()) :
                                    Otf(trackerType), Param(std::move(parameters)) {};

    [[nodiscard]] std::vector<Object> trackObjects(const cv::Mat& image, const std::vector<Object>& objectDetections);
    [[nodiscard]] size_t numberOfTrackedObjects() const;

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