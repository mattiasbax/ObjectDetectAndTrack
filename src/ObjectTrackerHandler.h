#pragma once
#include <memory>
#include <opencv2/video/tracking.hpp>

class ObjectTrackerHandler {
public:
    struct Parameters
    {
    };
    explicit ObjectTrackerHandler(const cv::Ptr<cv::Tracker> tracker, const Parameters& parameters) : Tracker(tracker), Tracking(false) {};
    void startTracking(const cv::Mat& image, const cv::Rect& bbox);
    void stopTracking();
    cv::Rect update(const cv::Mat& image);
private:
    const cv::Ptr<cv::Tracker> Tracker;
    bool Tracking;
};