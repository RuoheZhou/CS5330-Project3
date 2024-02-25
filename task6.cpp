
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <map>
#include <fstream>
#include <sstream> 
#include "filters.hpp"

struct ObjectFeature {
    std::string label;
    std::vector<double> features;
};

std::vector<ObjectFeature> loadFeatureDatabase(const std::string& filename) {
    std::vector<ObjectFeature> database;
    std::ifstream file(filename);
    std::string line;
    std::getline(file, line); // Skip header

    while (std::getline(file, line)) {
        std::istringstream ss(line);
        ObjectFeature obj;
        std::getline(ss, obj.label, ','); // First entry is the label

        std::string feature;
        while (std::getline(ss, feature, ',')) {
            obj.features.push_back(std::stod(feature));
        }
        database.push_back(obj);
    }
    return database;
}

double euclideanDistance(const std::vector<double>& vec1, const std::vector<double>& vec2) {
    double distance = 0.0;
    for (size_t i = 0; i < vec1.size(); ++i) {
        distance += std::pow(vec1[i] - vec2[i], 2);
    }
    return std::sqrt(distance);
}

std::string findBestMatchingLabel(const std::vector<double>& features, const std::vector<ObjectFeature>& database) {
    double minDistance = std::numeric_limits<double>::max();
    std::string bestLabel = "Unknown";

    for (const auto& obj : database) {
        double distance = euclideanDistance(features, obj.features);
        if (distance < minDistance) {
            minDistance = distance;
            bestLabel = obj.label;
        }
    }

    return bestLabel;
}


int main() {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Unable to open video device" << std::endl;
        return -1;
    }

    cv::namedWindow("Segmented", cv::WINDOW_AUTOSIZE);

    // Load the feature database from the specified CSV file
    std::vector<ObjectFeature> database = loadFeatureDatabase("../data/features.csv");

    cv::Mat frame, thresholded, segmented, eroded, dilated;
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
            for (const auto& reg : prevRegions) {
                cv::Moments m = cv::moments(labels == reg.first, true); // Recompute moments for this region
                std::vector<double> features = {m.m00, m.m10, m.m01, m.m20, m.m11, m.m02, m.m30, m.m21, m.m12, m.m03};

                std::string bestMatchLabel = findBestMatchingLabel(features, database);
                std::cout << "Best Match: " << bestMatchLabel << std::endl;
            }
        } else {
            for (const auto& reg : prevRegions) {
                computeFeatures(frame, labels, reg.first, reg.second.centroid, reg.second.color);
            }
        }

        if (key == 'q' || key == 27) break;
        cv::imshow("Original Video", frame);
    }

    cv::destroyAllWindows();
    return 0;
}
