#include <opencv2/opencv.hpp>
#include <stdio.h>
#include<iostream>


int erosion(cv::Mat & src, cv::Mat & dst, int kernelSize, int connectedness);
int dilation(cv::Mat & src, cv::Mat & dst, int kernelSize, int connectedness);
int thresholding(cv::Mat & src, cv::Mat & dst, int kernelSize);
