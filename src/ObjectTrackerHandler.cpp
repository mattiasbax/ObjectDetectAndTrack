#include "ObjectTrackerHandler.h"
#include <iostream>

namespace
{
}

void ObjectTrackerHandler::createNewTracker(const cv::Mat& image, const ObjectDetection& object)
{
    if (TrackedObjects.size() < Param.MaxNumberOfTrackedObjects)
    {
        if (std::find(Param.ClassesToTrack.begin(), Param.ClassesToTrack.end(), object.classId) != Param.ClassesToTrack.end())
        {
            std::cout << "Creating a new tracker" << std::endl;
            TrackedObjects.push_back({Otf.createTracker(), object});
            TrackedObjects.back().Tracker->init(image, object.boundingBox);
        }
    }
}


bool ObjectTrackerHandler::isObjectAlreadyTracked(const ObjectDetection& object) const
{
    bool isAlreadyTracked = false;
    for (const auto& trackedObject : TrackedObjects)
    {
        if (object.classId != trackedObject.Object.classId)
        {
            continue;
        }
        isAlreadyTracked = true;
    }
    return isAlreadyTracked;
}

std::vector<ObjectDetection> ObjectTrackerHandler::trackObjects(const cv::Mat& image, const std::vector<ObjectDetection>& objectDetections)
{
    for (const auto& objectDetection : objectDetections)
    {
        const bool isAlreadyTracked = isObjectAlreadyTracked(objectDetection);
        if (not isAlreadyTracked)
        {
            createNewTracker(image, objectDetection);
        }
    }

    std::vector<ObjectDetection> trackedObjects;
    for (auto& trackedObject : TrackedObjects)
    {
        const bool objectFound = trackedObject.Tracker->update(image, trackedObject.Object.boundingBox);
        if (objectFound)
        {
            trackedObjects.push_back(trackedObject.Object);
        }
    }

    return trackedObjects;
}
