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
    for (const auto& reg : prevRegions) {
        cv::Point2d prevCentroid = reg.second.centroid;
        double distance = cv::norm(centroid - prevCentroid);
        if (distance < 50.0) { // Threshold for matching centroids
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

void computeFeatures(cv::Mat &src, const cv::Mat &labels, int label, const cv::Point2d &centroid, const cv::Vec3b &color) {
    // Create a single-channel, binary mask for the current label
    cv::Mat mask = cv::Mat::zeros(labels.size(), CV_8U);
    for (int y = 0; y < labels.rows; ++y) {
        for (int x = 0; x < labels.cols; ++x) {
            if (labels.at<int>(y, x) == label) {
                mask.at<uchar>(y, x) = 255; // Set the pixel to white for the current label
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

    // Print out the calculated features
    std::cout << "Label: " << label << " | Angle: " << angle << " | Percent Filled: " << percentFilled << " | Aspect Ratio: " << aspectRatio << std::endl;

    // Draw the rotated rectangle on the source image
    cv::Point2f rectPoints[4];
    rotRect.points(rectPoints);
    for (int j = 0; j < 4; j++) {
        cv::line(src, rectPoints[j], rectPoints[(j + 1) % 4], cv::Scalar(color), 2);
    }
}


int thresholding(cv::Mat &src, cv::Mat &dst, int threshold) {
    cv::Mat grayscale_img;
    cv::cvtColor(src, grayscale_img, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(grayscale_img, grayscale_img, cv::Size(5, 5), 0);
    cv::threshold(grayscale_img, dst, threshold, 255, cv::THRESH_BINARY_INV);
    return 0;
}

int main() {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Unable to open video device" << std::endl;
        return -1;
    }

    cv::namedWindow("Original Video", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Segmented", cv::WINDOW_AUTOSIZE);

    cv::Mat frame, thresholded, segmented;
    std::map<int, RegionInfo> currentRegions;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        thresholding(frame, thresholded, 100);
        cleanAndSegment(thresholded, segmented, 500, currentRegions);

        for (const auto& reg : currentRegions) {
            computeFeatures(frame, segmented, reg.first, reg.second.centroid, reg.second.color);
        }

        cv::imshow("Original Video", frame);
        cv::imshow("Segmented", segmented);

        char key = static_cast<char>(cv::waitKey(10));
        if (key == 'q' || key == 27) { // 'q' or ESC key to exit
            break;
        }

        prevRegions = currentRegions;
    }

    return 0;
}
