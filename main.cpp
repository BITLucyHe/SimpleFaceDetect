#include "opencv2/core/base.hpp"
#include "opencv2/core/mat.hpp"
#include "opencv2/videoio.hpp"
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <fstream>

std::vector<std::string> classes;
std::vector<float> confidences;
std::vector<cv::Rect> boxes;
std::vector<int> classIds;
std::vector<int> indices;

void frameProcess(cv::Mat &frame, cv::dnn::Net &net, const std::vector<std::string> &classes, float confThreshold, float nmsThreshold) {
    cv::Mat blob;
    cv::dnn::blobFromImage(frame, blob, 1/255.0, cv::Size(320, 320), cv::Scalar(0, 0, 0), true, false);
    // std::cout << "blob size: " << blob.size << std::endl;

    net.setInput(blob);
    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());
    // std::cout << "output size: " << outputs[0].size << std::endl;

    for (int i = 0; i < outputs[0].size[1]; i ++) {
        float* data = (float*)outputs[0].data + i * outputs[0].size[2];

        if (data[4] < confThreshold) continue;

        float x_center = data[0] / 320.0f;
        float y_center = data[1] / 320.0f;
        float width = data[2] / 320.0f;
        float height = data[3] / 320.0f;
        float confidence = data[4];

        float* classes_scores = data + 5;
        cv::Mat scores(1, classes.size(), CV_32FC1);
        cv::Point classIdPoint;
        double maxClassScore;
        cv::minMaxLoc(scores, NULL, &maxClassScore, NULL, &classIdPoint);
        int class_id = classIdPoint.x;

        /********************************DEBUG USE********************************/
        // std::cout << "Detection " << i << ": " << std::endl;
        // std::cout << "  x_center: " << x_center << std::endl;
        // std::cout << "  y_center: " << y_center << std::endl;
        // std::cout << "  width: " << width << std::endl;
        // std::cout << "  height: " << height << std::endl;
        // std::cout << "  confidence: " << confidence << std::endl;
        // std::cout << "  class_id: " << class_id << std::endl;
        // std::cout << "  frame cols: " << frame.cols << std::endl;
        // std::cout << "  frame rows: " << frame.rows << std::endl;
        // std::cout << std::endl;
        /********************************DEBUG USE********************************/

        int leftTop_x = (int)((x_center - width / 2) * frame.cols);
        int leftTop_y = (int)((y_center - height / 2) * frame.rows);
        int rect_width = (int)(width * frame.cols);
        int rect_height = (int)(height * frame.rows);

        confidences.push_back(confidence);
        boxes.push_back(cv::Rect(leftTop_x, leftTop_y, rect_width, rect_height));
        classIds.push_back((int)class_id);
    }

    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    // std::cout << "indices size: " << indices.size() << std::endl;
}

void frameDisplay (bool is_display, cv::Mat &frame, bool isImage) {
    if (is_display) {
        for (int idx : indices) {
            cv::Rect box = boxes[idx];
            std::string label = classes[classIds[idx]] + ": " + cv::format("%.2f", confidences[idx]);
            cv::rectangle(frame, box, cv::Scalar(120, 83, 34), 4);
            int baseLine;
            cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 1.2, 2, &baseLine);
            cv::rectangle(frame, cv::Rect(box.x, box.y - labelSize.height - baseLine, labelSize.width, labelSize.height + baseLine), cv::Scalar(120, 83, 34), cv::FILLED);
            cv::putText(frame, label, cv::Point(box.x, box.y - baseLine / 2), cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 127, 235), 2, cv::LINE_4);
        }
        cv::imshow("Detection", frame);
        if (isImage) {
            cv::waitKey(0);
        } else {
            cv::waitKey(1);
        }
    }
}

std::string keys =
        "{help h usage| | Print help message.}"
        "{model m   | ../best.onnx   | Path to the .onnx model you trained / wanna use.}"
        "{device    | 0              | camera device number.}"
        "{input i   |                | Path to input image or video file. Skip this argument to capture frames from a camera.}"
        "{classes   | ../classes.txt | Path to the text file with names of classes.}"
        "{conf      | 0.5            | Confidence threshold.}"
        "{nms       | 0.4            | Non-Maximum Suppression threshold.}"
        "{display d | true           | Display the result on screen}";

int main(int argc, char** argv) {

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about(" Object Detection using YOLOv5 (.onnx) model.\n\nFor example:\n./main -i=your/file/path\n");
    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }
    CV_Assert(parser.has("model"));
    cv::String modelPath = parser.get<cv::String>("model");
    int deviceNum = parser.get<int>("device");
    cv::String inputPath = parser.get<cv::String>("input");
    cv::String classesPath = parser.get<cv::String>("classes");
    float confThreshold = parser.get<float>("conf");
    float nmsThreshold = parser.get<float>("nms");
    bool is_display = parser.get<bool>("display");

    std::ifstream ifs(classesPath);
    CV_Assert(ifs.is_open());
    std::string line;
    while (getline(ifs, line)) {
        classes.push_back(line);
    }

    cv::dnn::Net net = cv::dnn::readNetFromONNX(modelPath);
    CV_Assert(!net.empty());

    cv::VideoCapture cap;
    cv::Mat frame;
    bool isImage = false;

    if (!inputPath.empty()) {
        // judge if it's image or video
        std::string ext = inputPath.substr(inputPath.find_last_of(".") + 1);
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

        if (ext == "bmp" || ext == "dib" || ext == "jpeg" || ext == "jpg" ||
            ext == "jpe" || ext == "jp2" || ext == "png" || ext == "webp" ||
            ext == "avif" || ext == "pbm" || ext == "pgm" || ext == "ppm" ||
            ext == "pxm" || ext == "pnm" || ext == "pfm" || ext == "sr" ||
            ext == "ras" || ext == "tiff" || ext == "tif" || ext == "exr" ||
            ext == "hdr" || ext == "pic") {
            isImage = true;
            frame = cv::imread(inputPath);
            CV_Assert(!frame.empty());
        } else {
            cap.open(inputPath);
        }
    } else {
        cap.open(deviceNum);
    }

    CV_Assert(isImage || cap.isOpened());

    cv::TickMeter tm;

    if (isImage) {
        tm.start();
        frameProcess(frame, net, classes, confThreshold, nmsThreshold);
        tm.stop();

        std::cout << "process time: " << tm.getTimeMilli() << " ms" << std::endl;

        frameDisplay(is_display, frame, isImage);

        confidences.clear();
        boxes.clear();
        classIds.clear();
    } else {
        while (true) {
            CV_Assert(cap.read(frame));

            tm.start();
            frameProcess(frame, net, classes, confThreshold, nmsThreshold);
            tm.stop();

            frameDisplay(is_display, frame, isImage);

            std::cout << "process time: " << tm.getFPS() << std::endl;
            std::cout << "fps: " << 1000 / tm.getFPS() << std::endl;
            std::cout << std::endl;

            confidences.clear();
            boxes.clear();
            classIds.clear();
        }
    }
    return 0;
}