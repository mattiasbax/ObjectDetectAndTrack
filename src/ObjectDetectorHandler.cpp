#include "ObjectDetectorHandler.h"
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <string>
#include <fstream>
#include <filesystem>

namespace
{
    std::vector<Object> nonMaxSupressDetections(const float confThreshold, const float nmsThreshold, std::vector<Object>&& objectDetections)
    {
        // Map where each key represents a class and the vector the detection belonging to that class
        std::map<int, std::vector<size_t>> classSets;
        size_t counter = 0;
        for (const auto& objectDetection : objectDetections)
        {
            classSets[objectDetection.classId].push_back(counter);
            counter++;
        }
        std::vector<Object> objectDetectionsNMS;
        for (const auto& classSet : classSets)
        {
            const std::vector<size_t>& detectionIndices = classSet.second;
            std::vector<cv::Rect> localBoxes;
            std::vector<float> localConfidences;
            std::vector<size_t> localDetectionIndices;
            for (const auto& detectionIdx : detectionIndices)
            {
                localBoxes.push_back(objectDetections[detectionIdx].boundingBox);
                localConfidences.push_back(objectDetections[detectionIdx].confidence);
                localDetectionIndices.push_back(detectionIdx);
            }
            std::vector<int> nmsIndices;
            cv::dnn::NMSBoxes(localBoxes, localConfidences, confThreshold, nmsThreshold, nmsIndices);

            for (const auto& nmsIndex : nmsIndices)
            {
                objectDetectionsNMS.emplace_back(std::move(objectDetections[localDetectionIndices[nmsIndex]]));
            }
        }
        return objectDetectionsNMS;
    }


    std::vector<Object> postprocess(const float confThreshold, const float nmsThreshold, const cv::Mat& frame, const std::vector<cv::Mat>& outs, cv::dnn::Net& net, const std::vector<std::string>& classes)
    {
        std::vector<int> outLayers = net.getUnconnectedOutLayers();
        std::vector<Object> objectDetections;

        for (size_t i = 0; i < outs.size(); ++i)
        {
            // Network produces output blob with a shape NxC where N is a number of
            // detected objects and C is a number of classes + 4 where the first 4
            // numbers are [center_x, center_y, width, height]
            float* data = (float*)outs[i].data;
            for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
            {
                cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
                cv::Point classIdPoint;
                double confidence;
                minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
                if (confidence > confThreshold)
                {
                    int centerX = (int)(data[0] * frame.cols);
                    int centerY = (int)(data[1] * frame.rows);
                    int width = (int)(data[2] * frame.cols);
                    int height = (int)(data[3] * frame.rows);
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;

                    if (confidence >= confThreshold)
                    {
                        objectDetections.push_back({classIdPoint.x, "", confidence, cv::Rect(left, top, width, height)});
                    }
                }
            }
        }

        if (outLayers.size() > 1)
        {
            return nonMaxSupressDetections(confThreshold, nmsThreshold, std::move(objectDetections));
        }
        else
        {
            return objectDetections;
        }
    }
} // End of ano namespace


bool ObjectDetectorHandler::init()
{
    //TODO: Build proper factory/builder class to create different version and sizes of yolo into the object Yolo
    //      which holds input size, scaleFactor, version, class names and the net itself
    const std::string yoloPath = std::filesystem::path(__FILE__).remove_filename().string()+"yolo/";
    const std::string weightPath = yoloPath+YoloName+".weights";
    const std::string cfgPath = yoloPath+YoloName+".cfg";
    const std::string classFilePath = yoloPath+YoloName+"_classes.txt";
    Yolo = cv::dnn::readNet(weightPath, cfgPath);
    Yolo.setPreferableBackend(Param.preferredBackend);
    Yolo.setPreferableTarget(Param.preferredTarget);
    OutLayerNames = Yolo.getUnconnectedOutLayersNames();
    std::ifstream ifs(classFilePath.c_str());
    if (!ifs.is_open())
    {
        // TODO: Make a proper log macro
        std::cout << "ERROR: Cannot open file: " << classFilePath << std::endl;
        Initialized = false;
        return false;
    }
    ClassNames.clear();
    std::string line;
    while (std::getline(ifs, line))
    {
        ClassNames.push_back(line);
    }
    Initialized = true;
    return true;
}


std::vector<Object> ObjectDetectorHandler::detectObjects(const cv::Mat& image)
{
    if (not Initialized)
    {
        std::cout << "ERROR: Not initialized." << std::endl;
        return std::vector<Object>{};
    }
    if (image.empty())
    {
        std::cout << "ERROR: Cannot do detection on empty image." << std::endl;
        return std::vector<Object>{};
    }

    // Create a 4D blob from a frame.
    cv::Mat blob;
    cv::dnn::blobFromImage(image, blob, Param.ScaleFactor, Param.InputSize, Param.Mean, Param.SwapRGB);

    // Run a model.
    Yolo.setInput(blob);
    std::vector<cv::Mat> yoloOutput;
    Yolo.forward(yoloOutput, OutLayerNames);
    return postprocess(Param.ConfThreshold, Param.NmsThreshold, image, yoloOutput, Yolo, ClassNames);
}

const std::vector<std::string>& ObjectDetectorHandler::getClassNames() const
{
    return ClassNames;
}
