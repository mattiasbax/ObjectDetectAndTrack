#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/video/tracking.hpp>
//#include <opencv2/core/ocl.hpp>
#include <string>
#include "ObjectTrackerFactory.h"
#include "ObjectTrackerHandler.h"

int main() {
    // Activate the camera
    cv::VideoCapture camera(0);
    if (!camera.isOpened())
    {
        std::cout << "Cannot open camera feed" << std::endl;
        return 1;
    }
    constexpr int ESC = 27;
    cv::Mat cameraFrame;
    while (camera.read(cameraFrame))
    {
        cv::namedWindow("Camera Feed", cv::WindowFlags::WINDOW_AUTOSIZE); // Create a window for display.
        cv::imshow("Camera Feed", cameraFrame); // Show our image inside it.
        const int keyPress = cv::waitKey(1); // Wait for a keystroke in the window
        if (keyPress == ESC)
        {
            break;
        }
    }


    const ObjectTrackerFactory otf(ObjectTrackerFactory::TrackerType::KCF);
    ObjectTrackerHandler oth(otf.getTracker(), ObjectTrackerHandler::Parameters());

    //const std::string imgPath = "C:/Users/Mattias/CLionProjects/LearningOpenCV/src/images/starryNight.png";
    cv::Mat img;
    //img = cv::imread(imgPath, cv::ImreadModes::IMREAD_COLOR); // Read the file
    if (img.empty())
    {
        std::cout << "Could not read image" << std::endl;
    }
    else
    {
        cv::Rect2d bbox(250, 100, 200, 225);
        cv::rectangle(img, bbox, cv::Scalar(255, 255, 255), 2, 1);

        cv::namedWindow("Window", cv::WindowFlags::WINDOW_AUTOSIZE); // Create a window for display.
        cv::imshow("Window", img); // Show our image inside it.
        cv::waitKey(0); // Wait for a keystroke in the window
        //cv::imwrite("C:/Users/Mattias/CLionProjects/LearningOpenCV/src/images/starryNight.png", img);
    }
    return 0;
}
