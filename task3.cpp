#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <map>

struct RegionInfo {
    cv::Point2d centroid;
    cv::Vec3b color;
};

std::map<int, RegionInfo> prevRegions;

cv::Vec3b getColorForRegion(cv::Point2d centroid) {
    // If the region existed in the previous frame, try to use the same color
    for (const auto& reg : prevRegions) {
        cv::Point2d prevCentroid = reg.second.centroid;
        double distance = cv::norm(centroid - prevCentroid);

        if (distance < 50.0) { // Threshold for matching centroids (adjust as needed)
            return reg.second.color;
        }
    }

    // Otherwise, assign a new random color
    return cv::Vec3b(rand() % 256, rand() % 256, rand() % 256);
}

void cleanAndSegment(cv::Mat &src, cv::Mat &dst, int minRegionSize) {
    cv::Mat labels, stats, centroids;
    int nLabels = cv::connectedComponentsWithStats(src, labels, stats, centroids, 8, CV_32S);

    dst = cv::Mat::zeros(src.size(), CV_8UC3);
    std::map<int, RegionInfo> currentRegions;

    for (int i = 1; i < nLabels; i++) {
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        cv::Point2d centroid(centroids.at<double>(i, 0), centroids.at<double>(i, 1));

        if (area > minRegionSize) {
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

    // Update previous regions for the next frame
    prevRegions = std::move(currentRegions);
}

int thresholding(cv::Mat &src, cv::Mat &dst, int threshold) {
    cv::Mat grayscale_img;
    cv::cvtColor(src, grayscale_img, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(grayscale_img, grayscale_img, cv::Size(5, 5), 0);
    cv::threshold(grayscale_img, dst, threshold, 255, cv::THRESH_BINARY_INV);
    return 0;
}

int main() {
    cv::VideoCapture cap(0); // Adjust camera index as needed
    if (!cap.isOpened()) {
        std::cerr << "Error: Unable to open video device" << std::endl;
        return -1;
    }

    cv::namedWindow("Original Video", cv::WINDOW_NORMAL);
    cv::namedWindow("Segmented", cv::WINDOW_NORMAL);

    cv::Mat frame, thresholded, segmented;

    while (true) {
        cap >> frame; // Capture frame
        if (frame.empty()) break;

        // Thresholding to separate object from background
        thresholding(frame, thresholded, 100);

        // Clean up the image and segment into regions, ignoring small regions
        cleanAndSegment(thresholded, segmented, 500); // Adjust minRegionSize as needed

        // Display the original and segmented video
        cv::imshow("Original Video", frame);
        cv::imshow("Segmented", segmented);

        if (cv::waitKey(10) == 'q') break;
    }

    return 0;
}
