/**
 * @file filters.cpp
 * @author Ronak Bhanushali and Ruohe Zhou
 * @brief functions to help with object recognition
 * @date 2024-02-26
 * 
 */

#include "filters.hpp"

int thresholding(cv::Mat& src, cv::Mat& dst ,int threshold)
{
    // Convert image to grayscale
    cv::Mat grayscale_img;
    cv::cvtColor(src, grayscale_img, cv::COLOR_BGR2GRAY);
    
    // Apply Gaussian blur to reduce noise
    cv::GaussianBlur(grayscale_img, grayscale_img, cv::Size(5, 5), 0);
    
    // Create temporary image for thresholding
    cv::Mat temp = cv::Mat::zeros(src.size(), CV_8U);

    // Loop through each pixel
    for (int i = 0; i < src.rows; i++)
    {
        for (int j = 0; j < src.cols; j++)
        {
            // Get pixel value
            uchar pixel_value = grayscale_img.at<uchar>(i, j);
            
            // Thresholding
            if (pixel_value > threshold) {
                temp.at<uchar>(i, j) = 0;
            }
            else {
                temp.at<uchar>(i, j) = 255;
            }
        }
    }
    
    // Copy temporary image to output
    dst = temp.clone();
    
    // Return success
    return 0;
}

int erosion(cv::Mat & src, cv::Mat & dst, int kernelSize, int connectedness)
{
    // Get the number of rows and columns in the source image
    int num_rows = src.rows;
    int num_cols = src.cols;

    // Initialize the destination image with zeros
    dst = cv::Mat::zeros(src.size(), src.type());

    // Calculate the radius of the kernel
    int m = kernelSize / 2;

    // Perform erosion based on the type of connectivity
    if (connectedness == 8)
    {
        // Iterate over each pixel in the source image
        for (int i = m; i < num_rows - m; i++)
        {
            for (int j = m; j < num_cols - m; j++)
            {
                // Initialize the minimum value to maximum
                uchar min_val = 255;

                // Iterate over the kernel region
                for (int k = -m; k <= m; k++)
                {
                    for (int l = -m; l <= m; l++)
                    {
                        // Get the pixel value from the source image
                        uchar pixel = src.at<uchar>(i + k, j + l);

                        // Update the minimum value if the pixel value is smaller
                        if (pixel < min_val)
                            min_val = pixel;
                    }
                }

                // Assign the minimum value to the destination image
                dst.at<uchar>(i, j) = min_val;
            }
        }
    }
    else if (connectedness == 4)
    {
        // Iterate over each pixel in the source image
        for (int i = m; i < num_rows - m; i++)
        {
            for (int j = m; j < num_cols - m; j++)
            {
                // Initialize the minimum value to maximum
                uchar min_val = 255;

                // Iterate over the horizontal and vertical lines in the kernel
                for (int k = -m; k <= m; k++)
                {
                    // Get the pixel value from the source image along the horizontal line
                    uchar pixel = src.at<uchar>(i + k, j);
                    
                    // Update the minimum value if the pixel value is smaller
                    if (pixel < min_val)
                        min_val = pixel;
                }

                for (int l = -m; l <= m; l++)
                {
                    // Get the pixel value from the source image along the vertical line
                    uchar pixel = src.at<uchar>(i, j + l);
                    
                    // Update the minimum value if the pixel value is smaller
                    if (pixel < min_val)
                        min_val = pixel;
                }

                // Assign the minimum value to the destination image
                dst.at<uchar>(i, j) = min_val;
            }
        }
    }
    else
    {
        // Print an error message if the connectedness is neither 4 nor 8
        std::cout << "Connectedness can only be 4 or 8" << std::endl;
        return -1;
    }
    return 0;
}

int dilation(cv::Mat & src, cv::Mat & dst, int kernelSize, int connectedness)
{
    // Get the number of rows and columns in the source image
    int num_rows = src.rows;
    int num_cols = src.cols;

    // Initialize the destination image with zeros
    dst = cv::Mat::zeros(src.size(), src.type());

    // Calculate the radius of the kernel
    int m = kernelSize / 2;

    // Perform dilation based on the type of connectivity
    if (connectedness == 8)
    {
        // Iterate over each pixel in the source image
        for (int i = m; i < num_rows - m; i++)
        {
            for (int j = m; j < num_cols - m; j++)
            {
                // Initialize the maximum value to minimum
                uchar max_val = 0;

                // Iterate over the kernel region
                for (int k = -m; k <= m; k++)
                {
                    for (int l = -m; l <= m; l++)
                    {
                        // Get the pixel value from the source image
                        uchar pixel = src.at<uchar>(i + k, j + l);

                        // Update the maximum value if the pixel value is larger
                        if (pixel > max_val)
                            max_val = pixel;
                    }
                }

                // Assign the maximum value to the destination image
                dst.at<uchar>(i, j) = max_val;
            }
        }
    }
    else if (connectedness == 4)
    {
        // Iterate over each pixel in the source image
        for (int i = m; i < num_rows - m; i++)
        {
            for (int j = m; j < num_cols - m; j++)
            {
                // Initialize the maximum value to minimum
                uchar max_val = 0;

                // Iterate over the horizontal and vertical lines in the kernel
                for (int k = -m; k <= m; k++)
                {
                    // Get the pixel value from the source image along the horizontal line
                    uchar pixel = src.at<uchar>(i + k, j);
                    
                    // Update the maximum value if the pixel value is larger
                    if (pixel > max_val)
                        max_val = pixel;
                }

                for (int l = -m; l <= m; l++)
                {
                    // Get the pixel value from the source image along the vertical line
                    uchar pixel = src.at<uchar>(i, j + l);
                    
                    // Update the maximum value if the pixel value is larger
                    if (pixel > max_val)
                        max_val = pixel;
                }

                // Assign the maximum value to the destination image
                dst.at<uchar>(i, j) = max_val;
            }
        }
    }
    else
    {
        // Print an error message if the connectedness is neither 4 nor 8
        std::cout << "Connectedness can only be 4 or 8" << std::endl;
        return -1;
    }
    return 0;
}

// Function to get color for a region based on its centroid
cv::Vec3b getColorForRegion(cv::Point2d centroid, std::map<int, RegionInfo>& prevRegions) {
    // Iterate through previous regions
    for (const auto& reg : prevRegions) {
        // Get centroid of previous region
        cv::Point2d prevCentroid = reg.second.centroid;
        
        // Calculate distance between centroids
        double distance = cv::norm(centroid - prevCentroid);

        // If distance is within threshold, return color of previous region
        if (distance < 50) {
            return reg.second.color;
        }
    }

    // If no matching region found, return a random color
    return cv::Vec3b(rand() % 256, rand() % 256, rand() % 256);
}

// Function to segment objects in an image
cv::Mat segmentObjects(cv::Mat &src, cv::Mat &dst, int minRegionSize, std::map<int, RegionInfo>& prevRegions) {
    // Variables for connected components analysis
    cv::Mat labels, stats, centroids;
    int nLabels = cv::connectedComponentsWithStats(src, labels, stats, centroids, 8, CV_32S);

    // Initialize destination image
    dst = cv::Mat::zeros(src.size(), CV_8UC3);

    // Map to store current regions
    std::map<int, RegionInfo> currentRegions;

    // Iterate through labels
    for (int i = 1; i < nLabels; i++) {
        // Get area and centroid of region
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        cv::Point2d centroid(centroids.at<double>(i, 0), centroids.at<double>(i, 1));

        // Check if region meets minimum size requirement
        if (area > minRegionSize) {
            // Get color for region based on centroid
            cv::Vec3b color = getColorForRegion(centroid, prevRegions);
            
            // Add current region to map
            currentRegions[i] = {centroid, color};

            // Assign color to pixels of current region in destination image
            for (int y = 0; y < labels.rows; y++) {
                for (int x = 0; x < labels.cols; x++) {
                    if (labels.at<int>(y, x) == i) {
                        dst.at<cv::Vec3b>(y, x) = color;
                    }
                }
            }
        }
    }

    // Update previous regions with current regions
    prevRegions = std::move(currentRegions);
    
    // Return labels (connected components)
    return labels;
}

// Function to compute features for a region
cv::Moments computeFeatures(cv::Mat &src, const cv::Mat &labels, int label, const cv::Point2d &centroid, const cv::Vec3b &color) {
    // Create mask for the specified label
    cv::Mat mask = cv::Mat::zeros(labels.size(), CV_8U);
    for (int y = 0; y < labels.rows; ++y) {
        for (int x = 0; x < labels.cols; ++x) {
            if (labels.at<int>(y, x) == label) {
                mask.at<uchar>(y, x) = 255;
            }
        }
    }

    // Calculate moments of the mask
    cv::Moments m = cv::moments(mask, true);
    
    // Calculate orientation angle
    double angle = 0.5 * std::atan2(2 * m.mu11, m.mu20 - m.mu02);

    // Find minimum area rectangle enclosing the region
    std::vector<cv::Point> points;
    cv::findNonZero(mask, points);
    cv::RotatedRect rotRect = cv::minAreaRect(points);

    // Get corner points of the rectangle
    cv::Point2f rectPoints[4];
    rotRect.points(rectPoints);
    
    // Draw rectangle around the region
    for (int j = 0; j < 4; j++) {
        cv::line(src, rectPoints[j], rectPoints[(j + 1) % 4], cv::Scalar(color), 2);
    }

    // Get center and endpoint for drawing orientation line
    cv::Point center = rotRect.center;
    cv::Point endpoint(center.x + cos(angle) * 100, center.y + sin(angle) * 100);
    
    // Draw orientation line
    cv::line(src, center, endpoint, cv::Scalar(color), 2);

    // Return computed moments
    return m;
}
