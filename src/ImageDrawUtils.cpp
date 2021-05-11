#include "ImageDrawUtils.h"
#include <opencv2/imgproc.hpp>

namespace ImageDrawUtils
{

    void drawObjectsInImage(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame, const cv::Scalar& color, const std::vector<std::string>& classes)
    {
        rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 255, 0));

        std::string label = cv::format("%.2f", conf);
        if (!classes.empty())
        {
            CV_Assert(classId < (int)classes.size());
            label = classes[classId] + ": " + label;
        }

        int baseLine;
        cv::Size labelSize = getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        top = cv::max(top, labelSize.height);
        rectangle(frame, cv::Point(left, top - labelSize.height),
                  cv::Point(left + labelSize.width, top + baseLine), color, cv::FILLED);
        putText(frame, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar());
    }




}