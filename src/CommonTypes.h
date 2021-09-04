#pragma once
#include <opencv2/core.hpp>

struct Object {
    int classId;
    std::string className;
    double confidence;
    cv::Rect boundingBox;
};