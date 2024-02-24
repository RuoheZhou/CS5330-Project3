// #include <opencv2/opencv.hpp>
// #include <iostream>
// #include <vector>
// #include <map>
// #include "filters.hpp"


// void computeFeatures(cv::Mat &src, const cv::Mat &labels, int label, const cv::Point2d &centroid, const cv::Vec3b &color) {

//     cv::Mat mask = cv::Mat::zeros(labels.size(), CV_8U);
//     for (int y = 0; y < labels.rows; ++y) {
//         for (int x = 0; x < labels.cols; ++x) {
//             if (labels.at<int>(y, x) == label) {
//                 mask.at<uchar>(y, x) = 255;
//             }
//         }
//     }
//     // Calculate moments for the single-channel binary mask
//     cv::Moments m = cv::moments(mask, true);


//     // Calculate the angle of the axis of the least moment
//     double angle = 0.5 * std::atan2(2 * m.mu11, m.mu20 - m.mu02) * 180 / CV_PI;

//     // Calculate the oriented bounding box
//     std::vector<cv::Point> points;
//     cv::findNonZero(mask, points);
//     cv::RotatedRect rotRect = cv::minAreaRect(points);


//     // Draw the oriented bounding box
//     cv::Point2f rectPoints[4];
//     rotRect.points(rectPoints);
//     for (int j = 0; j < 4; j++) {
//         cv::line(src, rectPoints[j], rectPoints[(j + 1) % 4], cv::Scalar(color), 2);
//     }

//     // Draw the axis of the least moment
//     cv::Point center = rotRect.center;
//     cv::Point endpoint(center.x + cos(angle) * 100, center.y + sin(angle) * 100); // Arbitrary length of 100
//     cv::line(src, center, endpoint, cv::Scalar(color), 2);
// }

// int main() {
//     cv::VideoCapture cap(0);
//     if (!cap.isOpened()) {
//         std::cerr << "Error: Unable to open video device" << std::endl;
//         return -1;
//     }

//     // cv::namedWindow("Original Video", cv::WINDOW_AUTOSIZE);
//     cv::namedWindow("Segmented", cv::WINDOW_AUTOSIZE);

//     cv::Mat frame, thresholded, segmented, eroded, dilated;
//     std::map<int, RegionInfo> currentRegions;
//     std::map<int, RegionInfo> prevRegions;

//     while (true) {
//         cap >> frame;
//         if (frame.empty()) break;

//         // Thresholding to separate object from background
//         thresholding(frame, thresholded, 100);
//         dilation(thresholded,dilated,5,8);
//         erosion(dilated,eroded,5,4);

//         // Clean up the image and segment into regions, ignoring small regions
//         cv::Mat labels = cleanAndSegment(eroded, segmented, 500, prevRegions); // Adjust minRegionSize as needed
        
//         for (const auto& reg : prevRegions) {
//             computeFeatures(frame, labels, reg.first, reg.second.centroid, reg.second.color);
//         }
//         cv::imshow("Original Video", frame);
//         // cv::imshow("Segmented", segmented);

//         char key = static_cast<char>(cv::waitKey(10));
//         if (key == 'q' || key == 27) { // 'q' or ESC key to exit
//             cv::destroyAllWindows();
//             break;
//         }
//     }

//     return 0;
// }
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <map>
#include <fstream>
#include "filters.hpp"

// Overloaded function without file writing logic
void computeFeatures(cv::Mat &src, const cv::Mat &labels, int label, const cv::Point2d &centroid, const cv::Vec3b &color) {
    cv::Mat mask = cv::Mat::zeros(labels.size(), CV_8U);
    for (int y = 0; y < labels.rows; ++y) {
        for (int x = 0; x < labels.cols; ++x) {
            if (labels.at<int>(y, x) == label) {
                mask.at<uchar>(y, x) = 255;
            }
        }
    }

    cv::Moments m = cv::moments(mask, true);
    double angle = 0.5 * std::atan2(2 * m.mu11, m.mu20 - m.mu02) * 180 / CV_PI;

    std::vector<cv::Point> points;
    cv::findNonZero(mask, points);
    cv::RotatedRect rotRect = cv::minAreaRect(points);

    cv::Point2f rectPoints[4];
    rotRect.points(rectPoints);
    for (int j = 0; j < 4; j++) {
        cv::line(src, rectPoints[j], rectPoints[(j + 1) % 4], cv::Scalar(color), 2);
    }

    cv::Point center = rotRect.center;
    cv::Point endpoint(center.x + cos(angle) * 100, center.y + sin(angle) * 100);
    cv::line(src, center, endpoint, cv::Scalar(color), 2);
}

// Overloaded function with file writing logic
void computeFeatures(cv::Mat &src, const cv::Mat &labels, int label, const cv::Point2d &centroid, const cv::Vec3b &color, std::ofstream &file) {
    // Same content as the first overload
    computeFeatures(src, labels, label, centroid, color);

    // Additional file writing logic
    cv::Moments m = cv::moments(labels == label, true); // Recompute moments for file output
    if (file.is_open()) {
        file << label << "," << m.m00 << "," << m.m10 << "," << m.m01 << "," << m.m20 << "," << m.m11 << ",";
        file << m.m02 << "," << m.m30 << "," << m.m21 << "," << m.m12 << "," << m.m03 << std::endl;
    }
}

int main() {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Unable to open video device" << std::endl;
        return -1;
    }

    cv::namedWindow("Segmented", cv::WINDOW_AUTOSIZE);

    cv::Mat frame, thresholded, segmented, eroded, dilated;
    std::map<int, RegionInfo> currentRegions;
    std::map<int, RegionInfo> prevRegions;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        thresholding(frame, thresholded, 100);
        dilation(thresholded, dilated, 5, 8);
        erosion(dilated, eroded, 5, 4);

        cv::Mat labels = cleanAndSegment(eroded, segmented, 500, prevRegions);

        char key = static_cast<char>(cv::waitKey(10));
        if (key == 'N' || key == 'n') {
            std::string label;
            std::cout << "Enter a name/label for the moments data: ";
            std::cin >> label;

            std::ofstream file("../moments_data.csv", std::ios_base::app);
            if (file.is_open()) {
                file << "Label,m00,m10,m01,m20,m11,m02,m30,m21,m12,m03" << std::endl;
                for (const auto& reg : prevRegions) {
                    computeFeatures(frame, labels, reg.first, reg.second.centroid, reg.second.color, file);
                }
                file.close();
                std::cout << "Data saved successfully." << std::endl;
            } else {
                std::cerr << "Error opening file!" << std::endl;
            }
        } else {
            for (const auto& reg : prevRegions) {
                computeFeatures(frame, labels, reg.first, reg.second.centroid, reg.second.color);
            }
        }

        if (key == 'q' || key == 27) { // 'q' or ESC to quit
            break;
        }

        cv::imshow("Original Video", frame);
    }

    cv::destroyAllWindows();
    return 0;
}
