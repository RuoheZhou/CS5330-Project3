/**
 * @file task4.cpp
 * @author Ronak Bhanushali and Ruohe Zhou
 * @brief Task 4
 * @date 2024-02-26
 * 
 */
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <map>
#include <fstream>
#include "filters.hpp"

int main() {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Unable to open video device" << std::endl;
        return -1;
    }

    cv::namedWindow("Segmented", cv::WINDOW_AUTOSIZE);

    cv::Mat frame, thresholded, segmented, eroded, dilated;
    std::map<int, RegionInfo> prevRegions;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        thresholding(frame, thresholded, 100);
        dilation(thresholded, dilated, 5, 8);
        erosion(dilated, eroded, 5, 4);

        cv::Mat labels = segmentObjects(eroded, segmented, 500, prevRegions);

            for (const auto& reg : prevRegions) {
                computeFeatures(frame, labels, reg.first, reg.second.centroid, reg.second.color);
            }
        int key = cv::waitKey(10);
        if (key == 'q' || key == 27) { // 'q' or ESC to quit
            break;
        }

        cv::imshow("Original Video", frame);
    }

    cv::destroyAllWindows();
    return 0;
}
