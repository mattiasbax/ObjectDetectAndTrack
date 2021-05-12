#include "ImageDrawUtils.h"
#include <opencv2/imgproc.hpp>

namespace
{
        void drawClassInformationInImage(const Object& object, const std::vector<std::string>& classNames, cv::Mat& image, const cv::Scalar& color)
        {
            const auto& box = object.boundingBox;
            std::string label = cv::format("%.2f", object.confidence);
            if (!classNames.empty())
            {
                CV_Assert(object.classId < (int)classNames.size());
                label = classNames[object.classId] + ": " + label;
            }

            int baseLine;
            cv::Size labelSize = getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

            const float top = cv::max(box.y, labelSize.height);
            rectangle(image, cv::Point(box.x, top - labelSize.height),
                      cv::Point(box.x + labelSize.width, top + baseLine), color, cv::FILLED);
            putText(image, label, cv::Point(box.x, top), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar());
        }

    void drawIdentityInformationInImage(const Object& object, cv::Mat& image, const cv::Scalar& color)
    {
        const auto& box = object.boundingBox;

        int baseLine;
        cv::Size labelSize = getTextSize(object.identity, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        const float top = cv::max(box.y, labelSize.height);
        rectangle(image, cv::Point(box.x, top - labelSize.height),
                  cv::Point(box.x + labelSize.width, top + baseLine), color, cv::FILLED);
        putText(image, object.identity, cv::Point(box.x, top), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar());
    }
}


namespace ImageDrawUtils
{



    void drawObjectsInImage(const Object& object, const std::vector<std::string>& classNames, cv::Mat& image, const cv::Scalar& color,
                            ObjectType objectType)
    {
        const auto& box = object.boundingBox;
        rectangle(image, cv::Point(box.x, box.y),
                  cv::Point(box.x + box.width, box.y + box.height),
                  color);

        switch (objectType)
        {
            case ObjectType::Detection:
            {
                drawClassInformationInImage(object, classNames, image, color);
                break;
            }
            case ObjectType::Tracked:
            {
                drawIdentityInformationInImage(object, image, color);
                break;
            }
        }
    }




}