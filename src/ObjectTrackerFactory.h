#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/core/ocl.hpp>

class ObjectTrackerFactory
{
public:
    enum class TrackerType {
        MIL,
        GOTURN,
    };

    explicit ObjectTrackerFactory(TrackerType trackerType)
            : trackerType(trackerType) {}

    cv::Ptr<cv::Tracker> getTracker() const;
private:
    TrackerType trackerType;
};