#pragma once
#include <opencv2/core.hpp>
#include "CommonTypes.h"

namespace ImageDrawUtils
{
    enum class ObjectType
    {
        Detection,
        Tracked
    };

    void drawObjectsInImage(const Object& object, const std::vector<std::string>& classNames, cv::Mat& image, const cv::Scalar& color,
                            ObjectType objectType);
}