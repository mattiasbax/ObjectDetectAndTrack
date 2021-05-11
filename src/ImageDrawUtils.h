#pragma once
#include <opencv2/core.hpp>
namespace ImageDrawUtils
{
    void drawObjectsInImage(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame, const
                            cv::Scalar& color, const std::vector<std::string>& classes);
}