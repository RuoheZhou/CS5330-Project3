#include <opencv2/opencv.hpp>
#include <stdio.h>
#include<iostream>

struct RegionInfo {
    cv::Point2d centroid;
    cv::Vec3b color;
};

int erosion(cv::Mat & src, cv::Mat & dst, int kernelSize, int connectedness);
int dilation(cv::Mat & src, cv::Mat & dst, int kernelSize, int connectedness);
int thresholding(cv::Mat & src, cv::Mat & dst, int kernelSize);
cv::Mat segmentObjects(cv::Mat &src, cv::Mat &dst, int minRegionSize, std::map<int, RegionInfo>& prevRegions);
cv::Vec3b getColorForRegion(cv::Point2d centroid, std::map<int, RegionInfo>& prevRegions);
cv::Moments computeFeatures(cv::Mat &src, const cv::Mat &labels, int label, const cv::Point2d &centroid, const cv::Vec3b &color);