/**
 * @file task3.cpp
 * @author Ronak Bhanushali and Ruohe Zhou
 * @brief Task 3
 * @date 2024-02-26
 * 
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <map>
#include "filters.hpp"



int main() {
    cv::VideoCapture cap(0); // Adjust camera index as needed
    if (!cap.isOpened()) {
        std::cerr << "Error: Unable to open video device" << std::endl;
        return -1;
    }

    cv::namedWindow("Original Video", cv::WINDOW_NORMAL);
    cv::namedWindow("Segmented", cv::WINDOW_NORMAL);

    cv::Mat frame, thresholded, segmented, dilated, eroded;
    std::map<int, RegionInfo> prevRegions;

    while (true) {
        cap >> frame; // Capture frame
        if (frame.empty()) break;

        // Thresholding to separate object from background
        thresholding(frame, thresholded, 100);
        dilation(thresholded,dilated,5,8);
        erosion(dilated,eroded,5,4);

        // Clean up the image and segment into regions, ignoring small regions
        segmentObjects(eroded, segmented, 500, prevRegions); // Adjust minRegionSize as needed

        // Display the original and segmented video
        cv::imshow("Original Video", frame);
        cv::imshow("Segmented", segmented);

        if (cv::waitKey(10) == 'q') break;
    }

    return 0;
}
