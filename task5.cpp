#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <string>

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

void computeFeatures(const cv::Mat &src, const cv::Mat &labels, int label) {
    // Create a single-channel binary mask where the region corresponding to 'label' is white
    cv::Mat mask = cv::Mat::zeros(labels.size(), CV_8UC1);
    for (int y = 0; y < labels.rows; ++y) {
        for (int x = 0; x < labels.cols; ++x) {
            if (labels.at<int>(y, x) == label) {
                mask.at<uchar>(y, x) = 255;  // Set pixels to white for the region of interest
            }
        }
    }

    // Now 'mask' is guaranteed to be single-channel and can be safely used with cv::moments
    cv::Moments moments = cv::moments(mask, true);
    // Calculate central moments from raw moments
    double x_bar = moments.m10 / moments.m00;
    double y_bar = moments.m01 / moments.m00;
    double mu_11 = (moments.m11 - x_bar * moments.m01) / moments.m00;
    double mu_20 = (moments.m20 - x_bar * moments.m10) / moments.m00;
    double mu_02 = (moments.m02 - y_bar * moments.m01) / moments.m00;

    // Calculate orientation angle
    double theta = 0.5 * atan2(2 * mu_11, mu_20 - mu_02);

    // Find contours
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    

    // Calculate bounding rectangle
    cv::Rect bounding_rect = cv::boundingRect(contours[0]);  // Assuming only one contour exists

        // Rotate the bounding box
    cv::Point2f center(bounding_rect.x + bounding_rect.width / 2.0,
                    bounding_rect.y + bounding_rect.height / 2.0);
    cv::Mat rotation_matrix = cv::getRotationMatrix2D(center, theta * 180.0 / CV_PI, 1.0);
    std::vector<cv::Point2f> rotated_rect(4);
    cv::RotatedRect rotated_rect_info = cv::minAreaRect(contours[0]);
    rotated_rect_info.points(rotated_rect.data());

    // Convert each point to homogeneous coordinates and apply the transformation
    cv::Mat rotated_rect_mat(3, 4, CV_64FC1); // Homogeneous coordinates
    for (int i = 0; i < 4; ++i) {
        rotated_rect_mat.at<double>(0, i) = rotated_rect[i].x;
        rotated_rect_mat.at<double>(1, i) = rotated_rect[i].y;
        rotated_rect_mat.at<double>(2, i) = 1;
    }
    cv::Mat transformed_rect = rotation_matrix * rotated_rect_mat;

    // Convert the transformed homogeneous coordinates back to 2D points
    for (int i = 0; i < 4; ++i) {
        transformed_rect.col(i) /= transformed_rect.at<double>(2, i); // Normalize homogeneous coordinates
        rotated_rect[i].x = static_cast<float>(transformed_rect.at<double>(0, i));
        rotated_rect[i].y = static_cast<float>(transformed_rect.at<double>(1, i));
    }


    for (int i = 0; i < 4; ++i) {
        cv::line(labels, rotated_rect[i], rotated_rect[(i + 1) % 4], cv::Scalar(0, 255, 0), 2);
    }
   
}


void saveTrainingData(const std::vector<ObjectFeatures>& data) {
    std::ofstream outFile("../training_data.txt");
    for (const auto& item : data) {
        outFile << item.label << "," << item.angle << "," << item.percentFilled << "," << item.aspectRatio << std::endl;
    }
}

int main() {
    cv::VideoCapture cap(1);  // Adjust the device ID as needed
    if (!cap.isOpened()) {
        std::cerr << "Error: Unable to open video device." << std::endl;
        return -1;
    }

    std::vector<std::string> labels = {"Label1", "Label2", "Label3"};  // Predefined labels
    int currentLabelIndex = 0;  // Index to keep track of the current label

    cv::namedWindow("Original Video", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Segmented", cv::WINDOW_AUTOSIZE);

    cv::Mat frame, thresholded, segmented;
    std::map<int, RegionInfo> currentRegions;
    std::vector<ObjectFeatures> trainingData;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        cv::cvtColor(frame, thresholded, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(thresholded, thresholded, cv::Size(5, 5), 0);
        cv::threshold(thresholded, thresholded, 100, 255, cv::THRESH_BINARY_INV);

        cleanAndSegment(thresholded, segmented, 500, currentRegions);  // Adjust minRegionSize as needed

        cv::imshow("Original Video", frame);
        cv::imshow("Segmented", segmented);

        //char key = static_cast<char>(cv::waitKey(10));
        int key = cv::waitKey(30); 
        std::cout << key << std::endl;
        if (key == 78 || key == 110) {
            std::string currentLabel = labels[currentLabelIndex];  // Use the current label
            for (const auto& reg : currentRegions) {
                computeFeatures(frame, segmented, reg.first);
                // features.label = currentLabel;  // Assign the current label
                // trainingData.push_back(features);
            }
            std::cout << "Features labeled with: " << currentLabel << std::endl;
            currentLabelIndex = (currentLabelIndex + 1) % labels.size();  // Move to the next label
        } else if (key == 113 || key == 27) {  // 'q' or ESC to exit
            break;
        }

        prevRegions = currentRegions;
    }

    saveTrainingData(trainingData);
    std::cout << "Training data saved." << std::endl;

    return 0;
}
