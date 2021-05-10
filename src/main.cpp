#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string>
#include "ObjectTrackerHandler.h"
#include "ObjectDetectorHandler.h"
#include "CommonTypes.h"

void drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame, const std::vector<std::string>& classes)
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
              cv::Point(left + labelSize.width, top + baseLine), cv::Scalar::all(255), cv::FILLED);
    putText(frame, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar());
}

int main() {
    // Activate the camera
    cv::VideoCapture camera(0);
    if (!camera.isOpened())
    {
        std::cout << "Cannot open camera feed!" << std::endl;
        return 1;
    }

    constexpr int ESC = 27;
    static const std::string kWinName = "Object Detection and Tracking demo";
    namedWindow(kWinName, cv::WINDOW_AUTOSIZE);


    ObjectDetectorHandler odh("yolov4");
    ObjectTrackerHandler oth(ObjectTrackerFactory::TrackerType::CSRT);

    odh.init();
    const std::vector<std::string>& classNames = odh.getClassNames();

    // Frame loop
    cv::Mat frame;
    while (camera.read(frame) && (cv::waitKey(1) != ESC))
    {
        const std::vector<ObjectDetection> objectDetections = odh.detectObjects(frame);
        //const std::vector<ObjectDetection> trackedObjects = oth.trackObjects(objectDetections);

        for (const auto& objectDetection : objectDetections)
        {
            const cv::Rect& box = objectDetection.boundingBox;
            drawPred(objectDetection.classId, objectDetection.confidence, box.x, box.y,
            box.x + box.width, box.y + box.height, frame, classNames);
        }

        imshow(kWinName, frame);
    }

    // Close all windows
    cv::destroyAllWindows();
    return 0;
}
