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
        const std::vector<ObjectDetection> objectDetections = odh.detectObjects(frame);
        const std::vector<ObjectDetection> trackedObjects = oth.trackObjects(frame, objectDetections);

        for (const auto& objectDetection : objectDetections)
        {
            const cv::Rect& box = objectDetection.boundingBox;
            ImageDrawUtils::drawObjectsInImage(objectDetection.classId, static_cast<float>(objectDetection.confidence), box.x, box.y,
            box.x + box.width, box.y + box.height, frame, {255,0,0}, classNames);
        }

        for (const auto& trackedObject : trackedObjects)
        {
            const cv::Rect& box = trackedObject.boundingBox;
            ImageDrawUtils::drawObjectsInImage(trackedObject.classId, static_cast<float>(trackedObject.confidence), box.x, box.y,
                     box.x + box.width, box.y + box.height, frame, {0,255,0}, classNames);
        }


        // TODO: Draw helper class
        // TODO: Filter on only selected classes
        // TODO: Thread the detector
        // TODO: Smaller Yolo net
        imshow(kWinName, frame);
    }

    // Close all windows
    cv::destroyAllWindows();
    return 0;
}
