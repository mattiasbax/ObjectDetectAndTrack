#include "ObjectTrackerHandler.h"
#include <iostream>

namespace
{
}

void ObjectTrackerHandler::createNewTracker(const cv::Mat& image, const Object& detectedObject)
{
    if (TrackedObjects.size() < Param.MaxNumberOfTrackedObjects)
    {
        if (std::find(Param.ClassesToTrack.begin(), Param.ClassesToTrack.end(), detectedObject.classId) != Param.ClassesToTrack.end())
        {
            std::cout << "Creating a new tracker" << std::endl;
            Object trackedObject = detectedObject;
            trackedObject.identity = "Mattias";
            TrackedObjects.push_back({Otf.createTracker(), trackedObject});
            TrackedObjects.back().Tracker->init(image, trackedObject.boundingBox);
        }
    }
}


bool ObjectTrackerHandler::isObjectAlreadyTracked(const Object& object) const
{
    bool isAlreadyTracked = false;
    for (const auto& trackedObject : TrackedObjects)
    {
        if (object.classId != trackedObject.TrackedObject.classId)
        {
            continue;
        }
        isAlreadyTracked = true;
    }
    return isAlreadyTracked;
}

std::vector<Object> ObjectTrackerHandler::trackObjects(const cv::Mat& image, const std::vector<Object>& objectDetections)
{
    for (const auto& objectDetection : objectDetections)
    {
        const bool isAlreadyTracked = isObjectAlreadyTracked(objectDetection);
        if (not isAlreadyTracked)
        {
            createNewTracker(image, objectDetection);
        }
    }

    std::vector<Object> trackedObjects;
    for (auto& trackedObject : TrackedObjects)
    {
        const bool objectFound = trackedObject.Tracker->update(image, trackedObject.TrackedObject.boundingBox);
        if (objectFound)
        {
            trackedObjects.push_back(trackedObject.TrackedObject);
        }
    }

    return trackedObjects;
}
