#include <iostream>
#include "ObjectTrackerFactory.h"

cv::Ptr<cv::Tracker> ObjectTrackerFactory::getTracker() const
{
    cv::Ptr<cv::Tracker> tracker;
    switch (trackerType)
    {
        case TrackerType::MIL:
            tracker = cv::TrackerMIL::create();
            break;
        case TrackerType::GOTURN:
            tracker = cv::TrackerGOTURN::create();
            break;
        default:
            break;
    }
    return tracker;
}