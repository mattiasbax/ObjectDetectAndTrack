#include "ObjectTrackerHandler.h"
#include <iostream>

void ObjectTrackerHandler::startTracking(const cv::Mat& image, const cv::Rect& bbox)
{
    Tracker->init(image, bbox);
    Tracking = true;
}

void ObjectTrackerHandler::stopTracking()
{
    Tracking = false;
}

cv::Rect ObjectTrackerHandler::update(const cv::Mat& image)
{
    cv::Rect2d bbox;
    if (not Tracking)
    {
        std::cout << "Not currently tracking an object." << std::endl;
        return bbox;
    }
    Tracker->update(image, bbox);
    return bbox;
}