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
cv::Mat cleanAndSegment(cv::Mat &src, cv::Mat &dst, int minRegionSize, std::map<int, RegionInfo>& prevRegions);
cv::Vec3b getColorForRegion(cv::Point2d centroid, std::map<int, RegionInfo>& prevRegions);