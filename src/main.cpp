#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <fstream>
#include <string>
#include <thread>
#include <filesystem>
#include "ObjectTrackerFactory.h"
#include "ObjectTrackerHandler.h"

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

struct ObjectDetection
{
    int classId;
    double confidence;
    cv::Rect boundingBox;
};

void postprocess(cv::Mat& frame, const std::vector<cv::Mat>& outs, cv::dnn::Net& net, int backend, const std::vector<std::string>& classes)
{
    constexpr float confThreshold = 0.75;
    constexpr float nmsThreshold = 0.4;
    static std::vector<int> outLayers = net.getUnconnectedOutLayers();
    static std::string outLayerType = net.getLayer(outLayers[0])->type;

    std::vector<ObjectDetection> objectDetections;
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
                    objectDetections.push_back({classIdPoint.x, confidence, cv::Rect(left, top, width, height)});
                }
            }
        }
    }

    // Non Max Suppression
    std::vector<ObjectDetection> objectDetectionsNMS;
    if (outLayers.size() > 1)
    {
        // Map where each key represents a class and the vector the detection belonging to that class
        std::map<int, std::vector<size_t>> classSets;
        size_t counter = 0;
        for (const auto& objectDetection : objectDetections)
        {
            classSets[objectDetection.classId].push_back(counter);
            counter++;
        }
        for (const auto& classSet : classSets)
        {
            const int classIdx = classSet.first;
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
    }
    for (const auto& objectDetection : objectDetectionsNMS)
    {
        const cv::Rect& box = objectDetection.boundingBox;
        drawPred(objectDetection.classId, objectDetection.confidence, box.x, box.y,
                 box.x + box.width, box.y + box.height, frame, classes);
    }
}

template <typename T>
class QueueFPS : public std::queue<T>
{
public:
    QueueFPS() : counter(0) {}

    void push(const T& entry)
    {
        std::lock_guard<std::mutex> lock(mutex);

        std::queue<T>::push(entry);
        counter += 1;
        if (counter == 1)
        {
            // Start counting from a second frame (warmup).
            tm.reset();
            tm.start();
        }
    }

    T get()
    {
        std::lock_guard<std::mutex> lock(mutex);
        T entry = this->front();
        this->pop();
        return entry;
    }

    float getFPS()
    {
        tm.stop();
        double fps = counter / tm.getTimeSec();
        tm.start();
        return static_cast<float>(fps);
    }

    void clear()
    {
        std::lock_guard<std::mutex> lock(mutex);
        while (!this->empty())
            this->pop();
    }

    unsigned int counter;

private:
    cv::TickMeter tm;
    std::mutex mutex;
};


int main() {
    // Setup the yolo network
    constexpr auto prefTarget = cv::dnn::DNN_TARGET_CUDA;
    constexpr auto prefBackend = cv::dnn::DNN_BACKEND_CUDA;
    const std::string yoloNet = "yolov4";
    const std::string yoloPath = std::filesystem::path(__FILE__).remove_filename().string()+"yolo/";
    const std::string weightPath = yoloPath+yoloNet+".weights";
    const std::string cfgPath = yoloPath+yoloNet+".cfg";
    const std::string classFilePath = yoloPath+yoloNet+"_classes.txt";
    cv::dnn::Net yolo = cv::dnn::readNet(weightPath, cfgPath);
    yolo.setPreferableBackend(prefBackend);
    yolo.setPreferableTarget(prefTarget);

    std::ifstream ifs(classFilePath.c_str());
    if (!ifs.is_open())
    {
        CV_Error(cv::Error::StsError, "File " + classFilePath + " not found");
    }
    std::string line;
    std::vector<std::string> classNames;
    while (std::getline(ifs, line))
    {
        classNames.push_back(line);
    }


    // Activate the camera
    cv::VideoCapture camera(0);
    if (!camera.isOpened())
    {
        std::cout << "Cannot open camera feed!" << std::endl;
        return 1;
    }


    // Create a window
    static const std::string kWinName = "Deep learning object detection in OpenCV";
    namedWindow(kWinName, cv::WINDOW_NORMAL);

    bool process = true;
    // Frames capturing thread
    QueueFPS<cv::Mat> framesQueue;
    std::thread framesThread([&](){
        cv::Mat frame;
        while (process)
        {
            camera >> frame;
            if (!frame.empty())
                framesQueue.push(frame.clone());
            else
                break;
        }
    });

    // Frames processing thread
    //TODO: Fix a map<enum class sizeType, cv::Size()> for e.g. sizeType::416 -> cv::Size(416,416)
    constexpr int inpWidth = 416;
    constexpr int inpHeight = 416;
    constexpr float scaleFactor = 1./255.;
    constexpr bool swapRB = true;
    const std::vector<std::string> outNames = yolo.getUnconnectedOutLayersNames();
    QueueFPS<cv::Mat> processedFramesQueue;
    QueueFPS<std::vector<cv::Mat> > predictionsQueue;
    std::thread processingThread([&](){
        cv::Mat blob;
        while (process)
        {
            // Get a next frame
            cv::Mat frame;
            {
                if (!framesQueue.empty())
                {
                    frame = framesQueue.get();
                    framesQueue.clear();  // Skip the rest of frames
                }
            }

            // Process the frame
            if (!frame.empty())
            {
                // Create a 4D blob from a frame.
                static cv::Mat blob;
                cv::dnn::blobFromImage(frame, blob, scaleFactor, cv::Size(inpWidth,inpHeight),
                                       cv::Scalar(0.,0.,0.), swapRB, false);
                // Run a model.
                yolo.setInput(blob);
                processedFramesQueue.push(frame);

                std::vector<cv::Mat> outs;
                yolo.forward(outs, outNames);
                predictionsQueue.push(outs);
            }
        }
    });

    // Postprocessing and rendering loop
    constexpr int ESC = 27;
    while (cv::waitKey(1) != ESC)
    {
        if (predictionsQueue.empty())
        {
            continue;
        }

        std::vector<cv::Mat> outs = predictionsQueue.get();
        cv::Mat frame = processedFramesQueue.get();

        postprocess(frame, outs, yolo, prefBackend, classNames);

        if (predictionsQueue.counter > 1)
        {
            std::string label = cv::format("Camera: %.2f FPS", framesQueue.getFPS());
            putText(frame, label, cv::Point(0, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0));

            label = cv::format("Network: %.2f FPS", predictionsQueue.getFPS());
            putText(frame, label, cv::Point(0, 30), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0));

            label = cv::format("Skipped frames: %d", framesQueue.counter - predictionsQueue.counter);
            putText(frame, label, cv::Point(0, 45), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0));
        }
        imshow(kWinName, frame);
    }

    process = false;
    framesThread.join();
    processingThread.join();


    //const ObjectTrackerFactory otf(ObjectTrackerFactory::TrackerType::KCF);
    //ObjectTrackerHandler oth(otf.getTracker(), ObjectTrackerHandler::Parameters());

    // Frame loop
    //while (camera.read(frame) && (cv::waitKey(1) != ESC))
    //{

    //}

    // Close all windows
    cv::destroyAllWindows();
    return 0;
}
