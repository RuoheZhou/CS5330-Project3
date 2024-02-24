#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <map>
#include "filters.hpp"


void computeFeatures(cv::Mat &src, const cv::Mat &labels, int label, const cv::Point2d &centroid, const cv::Vec3b &color) {
   
    cv::Mat mask = cv::Mat::zeros(labels.size(), CV_8U);
    for (int y = 0; y < labels.rows; ++y) {
        for (int x = 0; x < labels.cols; ++x) {
            if (labels.at<int>(y, x) == label) {
                mask.at<uchar>(y, x) = 255;
            }
        }
    }
    // Calculate moments for the single-channel binary mask
    cv::Moments m = cv::moments(mask, true);

    // Calculate the angle of the axis of the least moment
    double angle = 0.5 * std::atan2(2 * m.mu11, m.mu20 - m.mu02) * 180 / CV_PI;

    // Find non-zero points for the minAreaRect calculation
    std::vector<cv::Point> points;
    cv::findNonZero(mask, points);

    if (points.empty()) {
        std::cerr << "No points found for label " << label << std::endl;
        return; // Skip further processing if no points found
    }

    // Ensure points are of correct type for contourArea and minAreaRect
    std::vector<cv::Point2f> floatPoints;
    for (const auto& pt : points) {
        floatPoints.push_back(cv::Point2f(pt.x, pt.y));
    }

    cv::RotatedRect rotRect = cv::minAreaRect(floatPoints);

    // Calculate percent filled and aspect ratio
    double area = cv::contourArea(floatPoints);
    double rectArea = rotRect.size.width * rotRect.size.height;
    double percentFilled = area / rectArea;
    double aspectRatio = rotRect.size.width / rotRect.size.height;

    // Draw the rotated rectangle on the source image
    cv::Point2f rectPoints[4];
    rotRect.points(rectPoints);
    for (int j = 0; j < 4; j++) {
        cv::line(src, rectPoints[j], rectPoints[(j + 1) % 4], cv::Scalar(color), 2);
    }
}

int main() {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Unable to open video device" << std::endl;
        return -1;
    }

    cv::namedWindow("Original Video", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Segmented", cv::WINDOW_AUTOSIZE);

    cv::Mat frame, thresholded, segmented, eroded, dilated;
    std::map<int, RegionInfo> currentRegions;
    std::map<int, RegionInfo> prevRegions;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        // Thresholding to separate object from background
        thresholding(frame, thresholded, 150);
        dilation(thresholded,dilated,5,8);
        erosion(dilated,eroded,5,4);

        // Clean up the image and segment into regions, ignoring small regions
        cv::Mat labels = cleanAndSegment(eroded, segmented, 500, prevRegions); // Adjust minRegionSize as needed

        cv::imshow("Original Video", frame);
        cv::imshow("Segmented", segmented);
        
        for (const auto& reg : prevRegions) {
            computeFeatures(frame, labels, reg.first, reg.second.centroid, reg.second.color);
        }


        char key = static_cast<char>(cv::waitKey(10));
        if (key == 'q' || key == 27) { // 'q' or ESC key to exit
            cv::destroyAllWindows();
            break;
        }
    }

    return 0;
}
