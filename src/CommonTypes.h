#pragma once
#include <opencv2/core.hpp>

struct ObjectDetection
{
    int classId;
    double confidence;
    cv::Rect boundingBox;
};