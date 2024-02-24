#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <string>

using namespace cv;
using namespace std;

struct RegionInfo {
    cv::Point2d centroid;
    cv::Vec3b color;
};

struct ObjectFeatures {
    double angle;
    double percentFilled;
    double aspectRatio;
    std::string label;
};

std::map<int, RegionInfo> prevRegions;

cv::Vec3b getColorForRegion(cv::Point2d centroid) {
    for (const auto& reg : prevRegions) {
        cv::Point2d prevCentroid = reg.second.centroid;
        double distance = cv::norm(centroid - prevCentroid);
        if (distance < 50.0) {
            return reg.second.color;
        }
    }
    return cv::Vec3b(rand() % 256, rand() % 256, rand() % 256);
}

void cleanAndSegment(cv::Mat &src, cv::Mat &dst, int minRegionSize, std::map<int, RegionInfo> &currentRegions) {
    cv::Mat labels, stats, centroids;
    int nLabels = cv::connectedComponentsWithStats(src, labels, stats, centroids, 8, CV_32S);

    dst = cv::Mat::zeros(src.size(), CV_8UC3);

    for (int i = 1; i < nLabels; i++) {
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area > minRegionSize) {
            cv::Point2d centroid(centroids.at<double>(i, 0), centroids.at<double>(i, 1));
            cv::Vec3b color = getColorForRegion(centroid);
            currentRegions[i] = {centroid, color};
            for (int y = 0; y < labels.rows; y++) {
                for (int x = 0; x < labels.cols; x++) {
                    if (labels.at<int>(y, x) == i) {
                        dst.at<cv::Vec3b>(y, x) = color;
                    }
                }
            }
        }
    }
}

ObjectFeatures computeFeatures(const cv::Mat &src, const cv::Mat &labels, int label) {
    cv::Mat mask = cv::Mat::zeros(labels.size(), CV_8UC1);
    for (int y = 0; y < labels.rows; ++y) {
        for (int x = 0; x < labels.cols; ++x) {
            if (labels.at<int>(y, x) == label) {
                mask.at<uchar>(y, x) = 255;  // Set pixels to white for the region of interest
            }
        }
    }

    cv::Moments m = cv::moments(mask, true);
    double angle = 0.5 * atan2(2 * m.mu11, m.mu20 - m.mu02) * 180 / CV_PI;

    std::vector<cv::Point> points;
    cv::findNonZero(mask, points);
    if (points.empty()) return {0, 0, 0, ""};

    std::vector<cv::Point2f> floatPoints(points.begin(), points.end());
    cv::RotatedRect rotRect = cv::minAreaRect(floatPoints);

    double area = cv::contourArea(floatPoints);
    double rectArea = rotRect.size.width * rotRect.size.height;
    double percentFilled = area / rectArea;
    double aspectRatio = rotRect.size.width / rotRect.size.height;

    return {angle, percentFilled, aspectRatio, ""};
}

void saveTrainingData(const std::vector<ObjectFeatures>& data) {
    std::ofstream outFile("training_data.txt");
    for (const auto& item : data) {
        outFile << item.label << "," << item.angle << "," << item.percentFilled << "," << item.aspectRatio << std::endl;
    }
}

int main() {
    cv::VideoCapture cap(1);
    if (!cap.isOpened()) {
        std::cerr << "Error: Unable to open video device." << std::endl;
        return -1;
    }

    std::vector<std::string> labels = {"Label1", "Label2", "Label3"};
    int currentLabelIndex = 0;

    cv::namedWindow("Original Video", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Segmented", cv::WINDOW_AUTOSIZE);

    cv::Mat frame, thresholded, segmented;
    std::map<int, RegionInfo> currentRegions;
    std::vector<ObjectFeatures> trainingData;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        cv::cvtColor(frame, thresholded, cv::COLOR_BGR2GRAY);
        GaussianBlur(thresholded, thresholded, Size(5, 5), 0);
        adaptiveThreshold(thresholded, thresholded, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 11, 2);

        cleanAndSegment(thresholded, segmented, 500, currentRegions);

        cv::imshow("Original Video", frame);
        cv::imshow("Segmented", segmented);

        char key = static_cast<char>(cv::waitKey(10));
        if (key == 'N' || key == 'n') {
            std::string currentLabel = labels[currentLabelIndex];
            for (const auto& reg : currentRegions) {
                ObjectFeatures features = computeFeatures(frame, segmented, reg.first);
                features.label = currentLabel;
                trainingData.push_back(features);
            }
            std::cout << "Features labeled with: " << currentLabel << std::endl;
            currentLabelIndex = (currentLabelIndex + 1) % labels.size();
        } else if (key == 'q' || key == 27) {
            break;
        }

        prevRegions = currentRegions;
    }

    saveTrainingData(trainingData);
    std::cout << "Training data saved." << std::endl;

    return 0;
}
