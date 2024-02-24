

#include <opencv2/opencv.hpp>
#include <iostream>
#include <unordered_map>

// Function to segment regions in the image and maintain color consistency

void segmentRegions(const cv::Mat& thresholded, cv::Mat& regionMap, std::unordered_map<int, cv::Vec3b>& regionColors, std::unordered_map<int, cv::Point>& regionCentroids, int minRegionPixelsThreshold) {
    // Perform connected components analysis
    cv::Mat labels, stats, centroids;
    int numRegions = cv::connectedComponentsWithStats(thresholded, labels, stats, centroids);

    // Initialize region map with zeros
    regionMap = cv::Mat::zeros(thresholded.size(), CV_8UC3);

    // Define a color palette for visualizing regions
    std::vector<cv::Vec3b> colors(numRegions);
    for (int i = 1; i < numRegions; ++i) {
        // Generate a random color for each new region
        if (regionColors.find(i) == regionColors.end()) {
            regionColors[i] = cv::Vec3b(rand() & 255, rand() & 255, rand() & 255);
        }
        colors[i] = regionColors[i];

        // Discard small regions
        if (stats.at<int>(i, cv::CC_STAT_AREA) < minRegionPixelsThreshold) {
            colors[i] = cv::Vec3b(0, 0, 0); // Set color to black for small regions
        }
    }

    // Iterate through the labeled regions and visualize them with consistent colors
    for (int y = 0; y < regionMap.rows; ++y) {
        for (int x = 0; x < regionMap.cols; ++x) {
            int label = labels.at<int>(y, x);
            if (label > 0) {
                cv::Vec3b color = colors[label];
                regionMap.at<cv::Vec3b>(y, x) = color;

                // Track centroid locations
                regionCentroids[label] = cv::Point(centroids.at<double>(label, 0), centroids.at<double>(label, 1));
            }
        }
    }
}

int main() {
    cv::VideoCapture cap(0); // Open default camera
    if (!cap.isOpened()) {
        std::cerr << "Error: Unable to open video device" << std::endl;
        return -1;
    }

    cv::namedWindow("Original Video", cv::WINDOW_NORMAL);
    cv::namedWindow("Region Map", cv::WINDOW_NORMAL);

    cv::Mat frame, regionMap;
    std::unordered_map<int, cv::Vec3b> regionColors;
    std::unordered_map<int, cv::Point> regionCentroids;

    for (;;) {
        cap >> frame; // Capture frame

        if (frame.empty()) {
            std::cerr << "Error: Frame is empty" << std::endl;
            break;
        }

        // Convert frame to grayscale
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        // Apply GaussianBlur to smooth the image and reduce noise
        cv::GaussianBlur(gray, gray, cv::Size(5, 5), 0);

        // Threshold the image to create a binary mask
        cv::Mat thresholded;
        cv::threshold(gray, thresholded, 100, 255, cv::THRESH_BINARY_INV);
        int minRegionPixelsThreshold = 100;
        // Perform region segmentation and maintain color consistency
        segmentRegions(thresholded, regionMap, regionColors, regionCentroids, minRegionPixelsThreshold);

        // Display original video and region map
        cv::imshow("Original Video", frame);
        cv::imshow("Region Map", regionMap);

        char key = cv::waitKey(10);
        if (key == 'q') {
            break; // Quit if 'q' is pressed
        }
    }

    return 0;
}
