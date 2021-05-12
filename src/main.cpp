#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string>
#include "ObjectTrackerHandler.h"
#include "ObjectDetectorHandler.h"
#include "CommonTypes.h"
#include "ImageDrawUtils.h"

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
    ObjectTrackerHandler oth(ObjectTrackerFactory::TrackerType::KCF);

    odh.init();
    const std::vector<std::string>& classNames = odh.getClassNames();

    // Frame loop
    cv::Mat frame;
    while (camera.read(frame) && (cv::waitKey(1) != ESC))
    {
        //const std::vector<Object> objectDetections = (oth.numberOfTrackedObjects() == 0) ? odh.detectObjects(frame) : std::vector<Object>{};
        const std::vector<Object> objectDetections = odh.detectObjects(frame);
        const std::vector<Object> trackedObjects = oth.trackObjects(frame, objectDetections);

        for (const auto& objectDetection : objectDetections)
        {
            ImageDrawUtils::drawObjectsInImage(objectDetection, classNames, frame, {255,0,0}, ImageDrawUtils::ObjectType::Detection);
        }

        for (const auto& trackedObject : trackedObjects)
        {
            ImageDrawUtils::drawObjectsInImage(trackedObject, classNames, frame, {0,255,0}, ImageDrawUtils::ObjectType::Tracked);
        }

        // TODO: Filter on only selected classes
        // TODO: Thread the detector
        // TODO: Smaller Yolo net
        // TODO: Valgrind https://www.youtube.com/watch?v=3l0BQs2ThTo&ab_channel=C%E1%90%A9%E1%90%A9WeeklyWithJasonTurner
        // TODO: Callgrind https://waterprogramming.wordpress.com/2017/06/08/profiling-c-code-with-callgrind/
        imshow(kWinName, frame);
    }

    // Close all windows
    cv::destroyAllWindows();
    return 0;
}
