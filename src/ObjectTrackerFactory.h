#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/core/ocl.hpp>

class ObjectTrackerFactory
{
public:
    enum class TrackerType {
        MIL,
        KCF,
        GOTURN,
        CSRT,
    };

    explicit ObjectTrackerFactory(TrackerType trackerType)
            : trackerType(trackerType) {}

    cv::Ptr<cv::Tracker> getTracker() const;
private:
    TrackerType trackerType;
};