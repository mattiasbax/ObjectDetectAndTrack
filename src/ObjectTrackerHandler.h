#pragma once
#include <memory>
#include <opencv2/tracking/tracking.hpp>

class ObjectTrackerHandler {
public:
    struct Parameters
    {
    };
    explicit ObjectTrackerHandler(const cv::Ptr<cv::Tracker> tracker, const Parameters& parameters = {}) : Tracker(tracker), Param(parameters), Tracking(false) {};
    void startTracking(const cv::Mat& image, const cv::Rect& bbox);
    void stopTracking();
    cv::Rect update(const cv::Mat& image);
private:
    const cv::Ptr<cv::Tracker> Tracker;
    const Parameters Param;
    bool Tracking;
};