/**
 * Ronak Bhanushali and Ruohe Zhou
 * Spring 2024
 * @file filters.hpp
 * @brief This file contains functions for image processing and object segmentation.
 */

#include <opencv2/opencv.hpp>
#include <stdio.h>
#include<iostream>

/**
 * @brief Struct to store information about a segmented region.
 */
struct RegionInfo {
    cv::Point2d centroid; ///< Centroid of the region.
    cv::Vec3b color; ///< Color of the region.
};

/**
 * @brief Performs erosion operation on a binary image.
 * @param src Input binary image.
 * @param dst Output image after erosion.
 * @param kernelSize Size of the erosion kernel.
 * @param connectedness Type of connectivity (4 or 8).
 * @return Returns 0 on success, -1 on failure.
 */
int erosion(cv::Mat & src, cv::Mat & dst, int kernelSize, int connectedness);

/**
 * @brief Performs dilation operation on a binary image.
 * @param src Input binary image.
 * @param dst Output image after dilation.
 * @param kernelSize Size of the dilation kernel.
 * @param connectedness Type of connectivity (4 or 8).
 * @return Returns 0 on success, -1 on failure.
 */
int dilation(cv::Mat & src, cv::Mat & dst, int kernelSize, int connectedness);

/**
 * @brief Performs thresholding operation on an input image.
 * @param src Input image.
 * @param dst Output binary image after thresholding.
 * @param kernelSize Size of the thresholding kernel.
 * @return Returns 0 on success, -1 on failure.
 */
int thresholding(cv::Mat & src, cv::Mat & dst, int kernelSize);

/**
 * @brief Segments objects in the input image and returns the segmented image.
 * @param src Input image.
 * @param dst Output segmented image.
 * @param minRegionSize Minimum size of a region to be considered an object.
 * @param prevRegions Map containing information about previously segmented regions.
 * @return Returns the segmented image.
 */
cv::Mat segmentObjects(cv::Mat &src, cv::Mat &dst, int minRegionSize, std::map<int, RegionInfo>& prevRegions);

/**
 * @brief Retrieves the color for a segmented region based on its centroid.
 * @param centroid Centroid of the region.
 * @param prevRegions Map containing information about previously segmented regions.
 * @return Returns the color for the region.
 */
cv::Vec3b getColorForRegion(cv::Point2d centroid, std::map<int, RegionInfo>& prevRegions);

/**
 * @brief Computes features for a segmented region.
 * @param src Input image.
 * @param labels Image containing labeled regions.
 * @param label Label of the region for which features are to be computed.
 * @param centroid Centroid of the region.
 * @param color Color of the region.
 * @return Returns the computed features for the region.
 */
cv::Moments computeFeatures(cv::Mat &src, const cv::Mat &labels, int label, const cv::Point2d &centroid, const cv::Vec3b &color);