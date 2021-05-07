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

inline void preprocess(const cv::Mat& frame, cv::dnn::Net& net, cv::Size inpSize, float scale,
                       const cv::Scalar& mean, bool swapRB)
{
    static cv::Mat blob;
    // Create a 4D blob from a frame.
    if (inpSize.width <= 0)
    {
        inpSize.width = frame.cols;
    }
    if (inpSize.height <= 0)
    {
        inpSize.height = frame.rows;
    }
    cv::dnn::blobFromImage(frame, blob, 1.0, inpSize, cv::Scalar(), swapRB, false, CV_8U);

    // Run a model.
    net.setInput(blob, "", scale, mean);
    if (net.getLayer(0)->outputNameToIndex("im_info") != -1)  // Faster-RCNN or R-FCN
    {
        resize(frame, frame, inpSize);
        cv::Mat imInfo = (cv::Mat_<float>(1, 3) << inpSize.height, inpSize.width, 1.6f);
        net.setInput(imInfo, "im_info");
    }
}

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

void postprocess(cv::Mat& frame, const std::vector<cv::Mat>& outs, cv::dnn::Net& net, int backend, const std::vector<std::string>& classes)
{
    constexpr float confThreshold = 0.5;
    constexpr float nmsThreshold = 0.4;

    static std::vector<int> outLayers = net.getUnconnectedOutLayers();
    static std::string outLayerType = net.getLayer(outLayers[0])->type;

    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    if (outLayerType == "DetectionOutput")
    {
        // Network produces output blob with a shape 1x1xNx7 where N is a number of
        // detections and an every detection is a vector of values
        // [batchId, classId, confidence, left, top, right, bottom]
        CV_Assert(outs.size() > 0);
        for (size_t k = 0; k < outs.size(); k++)
        {
            float* data = (float*)outs[k].data;
            for (size_t i = 0; i < outs[k].total(); i += 7)
            {
                float confidence = data[i + 2];
                if (confidence > confThreshold)
                {
                    int left   = (int)data[i + 3];
                    int top    = (int)data[i + 4];
                    int right  = (int)data[i + 5];
                    int bottom = (int)data[i + 6];
                    int width  = right - left + 1;
                    int height = bottom - top + 1;
                    if (width <= 2 || height <= 2)
                    {
                        left   = (int)(data[i + 3] * frame.cols);
                        top    = (int)(data[i + 4] * frame.rows);
                        right  = (int)(data[i + 5] * frame.cols);
                        bottom = (int)(data[i + 6] * frame.rows);
                        width  = right - left + 1;
                        height = bottom - top + 1;
                    }
                    classIds.push_back((int)(data[i + 1]) - 1);  // Skip 0th background class id.
                    boxes.push_back(cv::Rect(left, top, width, height));
                    confidences.push_back(confidence);
                }
            }
        }
    }
    else if (outLayerType == "Region")
    {
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

                    classIds.push_back(classIdPoint.x);
                    confidences.push_back((float)confidence);
                    boxes.push_back(cv::Rect(left, top, width, height));
                }
            }
        }
    }
    else
    {
        CV_Error(cv::Error::StsNotImplemented, "Unknown output layer type: " + outLayerType);
    }

    // NMS is used inside Region layer only on DNN_BACKEND_OPENCV for another backends we need NMS in sample
    // or NMS is required if number of outputs > 1
    if (outLayers.size() > 1 || (outLayerType == "Region" && backend != cv::dnn::DNN_BACKEND_OPENCV))
    {
        std::map<int, std::vector<size_t> > class2indices;
        for (size_t i = 0; i < classIds.size(); i++)
        {
            if (confidences[i] >= confThreshold)
            {
                class2indices[classIds[i]].push_back(i);
            }
        }
        std::vector<cv::Rect> nmsBoxes;
        std::vector<float> nmsConfidences;
        std::vector<int> nmsClassIds;
        for (std::map<int, std::vector<size_t> >::iterator it = class2indices.begin(); it != class2indices.end(); ++it)
        {
            std::vector<cv::Rect> localBoxes;
            std::vector<float> localConfidences;
            std::vector<size_t> classIndices = it->second;
            for (size_t i = 0; i < classIndices.size(); i++)
            {
                localBoxes.push_back(boxes[classIndices[i]]);
                localConfidences.push_back(confidences[classIndices[i]]);
            }
            std::vector<int> nmsIndices;
            cv::dnn::NMSBoxes(localBoxes, localConfidences, confThreshold, nmsThreshold, nmsIndices);
            for (size_t i = 0; i < nmsIndices.size(); i++)
            {
                size_t idx = nmsIndices[i];
                nmsBoxes.push_back(localBoxes[idx]);
                nmsConfidences.push_back(localConfidences[idx]);
                nmsClassIds.push_back(it->first);
            }
        }
        boxes = nmsBoxes;
        classIds = nmsClassIds;
        confidences = nmsConfidences;
    }

    for (size_t idx = 0; idx < boxes.size(); ++idx)
    {
        cv::Rect box = boxes[idx];
        drawPred(classIds[idx], confidences[idx], box.x, box.y,
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
    const std::string yoloNet = "yolov3";
    const std::string yoloPath = std::filesystem::path(__FILE__).remove_filename().string()+"yolo/";
    const std::string weightPath = yoloPath+yoloNet+".weights";
    const std::string cfgPath = yoloPath+yoloNet+".cfg";
    const std::string classFilePath = yoloPath+yoloNet+"_classes.txt";
    cv::dnn::Net yolo = cv::dnn::readNet(weightPath, cfgPath);
    yolo.setPreferableBackend(prefBackend);
    yolo.setPreferableTarget(prefTarget);
    const std::vector<std::string> outNames = yolo.getUnconnectedOutLayersNames();


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
    constexpr size_t asyncNumReq = 0;
    constexpr int inpWidth = 416;
    constexpr int inpHeight = 416;
    constexpr float scaleFactor = 0.00392;
    constexpr bool swapRB = true;
    const cv::Scalar mean = {0., 0., 0.};
    QueueFPS<cv::Mat> processedFramesQueue;
    QueueFPS<std::vector<cv::Mat> > predictionsQueue;
    std::thread processingThread([&](){
        std::queue<cv::AsyncArray> futureOutputs;
        cv::Mat blob;
        while (process)
        {
            // Get a next frame
            cv::Mat frame;
            {
                if (!framesQueue.empty())
                {
                    frame = framesQueue.get();
                    if (asyncNumReq)
                    {
                        if (futureOutputs.size() == asyncNumReq)
                            frame = cv::Mat();
                    }
                    else
                        framesQueue.clear();  // Skip the rest of frames
                }
            }

            // Process the frame
            if (!frame.empty())
            {
                preprocess(frame, yolo, cv::Size(inpWidth, inpHeight), scaleFactor, mean, swapRB);
                processedFramesQueue.push(frame);

                if (asyncNumReq)
                {
                    futureOutputs.push(yolo.forwardAsync());
                }
                else
                {
                    std::vector<cv::Mat> outs;
                    yolo.forward(outs, outNames);
                    predictionsQueue.push(outs);
                }
            }

            while (!futureOutputs.empty() &&
                   futureOutputs.front().wait_for(std::chrono::seconds(0)))
            {
                cv::AsyncArray async_out = futureOutputs.front();
                futureOutputs.pop();
                cv::Mat out;
                async_out.get(out);
                predictionsQueue.push({out});
            }
        }
    });

    // Postprocessing and rendering loop
    constexpr int ESC = 27;
    while (cv::waitKey(1) != ESC)
    {
        if (predictionsQueue.empty())
            continue;

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
    //auto tracker = otf.getTracker();

    //constexpr int ENTER = 13;

    //cv::Mat frame;
    //cv::Rect2d bbox(250, 100, 200, 225);
    //bool init = false;
    //bool photo = false;

    // Frame loop
    //while (camera.read(frame) && (cv::waitKey(1) != ESC))
    //{
    //    imshow("Window", frame);
    //}

    // Close all windows
    cv::destroyAllWindows();
    return 0;
}
