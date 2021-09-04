#include <filesystem>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string>

#include "CommonTypes.h"
#include "ImageDrawUtils.h"
#include "ObjectDetectorHandler.h"
#include "ObjectTracker.h"

int main() {
    // Activate the capture
    // cv::VideoCapture capture(0);
    const std::string videoPath = std::filesystem::path(__FILE__).remove_filename().string() + "rain_blinking.mp4";
    cv::VideoCapture capture(videoPath);

    if (!capture.isOpened()) {
        std::cout << "Cannot open image stream!" << std::endl;
        return 1;
    }

    constexpr int ESC = 27;
    static const std::string kWinName = "Object Detection and Tracking demo";
    namedWindow(kWinName, cv::WINDOW_AUTOSIZE);

    ObjectDetectorHandler odh("yolov4");
    odh.init();

    // Frame loop
    cv::Mat frame;
    while (capture.read(frame) && (cv::waitKey(10) != ESC)) {
        const std::vector<Object> objectDetections = odh.detectObjects(frame);

        for (const auto &objectDetection : objectDetections) {
            ImageDrawUtils::drawObjectsInImage(objectDetection, frame, {255, 0, 0},
                                               ImageDrawUtils::ObjectType::Detection);
        }

        // TODO: Valgrind
        // https://www.youtube.com/watch?v=3l0BQs2ThTo&ab_channel=C%E1%90%A9%E1%90%A9WeeklyWithJasonTurner
        // TODO: Callgrind
        // https://waterprogramming.wordpress.com/2017/06/08/profiling-c-code-with-callgrind/

        imshow(kWinName, frame);
    }

    // Close all windows
    cv::destroyAllWindows();
    return 0;
}
