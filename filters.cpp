#include "filters.hpp"

int thresholding(cv::Mat& src, cv::Mat& dst ,int threshold)
{
    int num_rows = src.rows;
    int num_cols = src.cols;
    cv::Mat grayscale_img,dilated_img,eroded_img;
    cv::cvtColor(src, grayscale_img, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(grayscale_img, grayscale_img, cv::Size(5, 5), 0);
    
    cv::Mat temp = cv::Mat::zeros(src.size(), CV_8U);

    for (int i = 0; i < num_rows; i++)
    {
        for (int j = 0; j < num_cols; j++)
        {
            uchar pixel_value = grayscale_img.at<uchar>(i, j);
            if (pixel_value > threshold) {
                temp.at<uchar>(i, j) = 0;
            }
            else {
                temp.at<uchar>(i, j) = 255;
            }
        }
    }
    dst = temp.clone();
    return 0;
}

int erosion(cv::Mat & src, cv::Mat & dst, int kernelSize, int connectedness)
{
    int num_rows = src.rows;
    int num_cols = src.cols;

    dst = cv::Mat::zeros(src.size(), src.type());

    int m = kernelSize / 2;

    if (connectedness == 8)
    {
        for (int i = m; i < num_rows - m; i++)
        {
            for (int j = m; j < num_cols - m; j++)
            {
                uchar min_val = 255;

                for (int k = -m; k <= m; k++)
                {
                    for (int l = -m; l <= m; l++)
                    {
                        uchar pixel = src.at<uchar>(i + k, j + l);
                        if (pixel < min_val)
                            min_val = pixel;
                    }
                }

                dst.at<uchar>(i, j) = min_val;
            }
        }
    }
    else if (connectedness == 4)
    {
        for (int i = m; i < num_rows - m; i++)
        {
            for (int j = m; j < num_cols - m; j++)
            {
                uchar min_val = 255;

                for (int k = -m; k <= m; k++)
                {
                    uchar pixel = src.at<uchar>(i + k, j);
                    if (pixel < min_val)
                        min_val = pixel;
                }
                for (int l = -m; l <= m; l++)
                {
                    uchar pixel = src.at<uchar>(i, j + l);
                    if (pixel < min_val)
                        min_val = pixel;
                }

                dst.at<uchar>(i, j) = min_val;
            }
        }
    }
    else
    {
        std::cout << "Connectedness can only be 4 or 8" << std::endl;
        return -1;
    }
    return 0;
}

int dilation(cv::Mat & src, cv::Mat & dst, int kernelSize, int connectedness)
{
    int num_rows = src.rows;
    int num_cols = src.cols;

    dst = cv::Mat::zeros(src.size(), src.type());

    int m = kernelSize / 2;

    if (connectedness == 8){
    for (int i = m; i < num_rows - m; i++)
    {
        for (int j = m; j < num_cols - m; j++)
        {
            uchar max_val = 0;

            for (int k = -m; k <= m; k++)
            {
                for (int l = -m; l <= m; l++)
                {
                    uchar pixel = src.at<uchar>(i + k, j + l);
                    if (pixel > max_val)
                        max_val = pixel;
                }
            }

            dst.at<uchar>(i, j) = max_val;
        }
    }
    }
    else if (connectedness == 4)
    {
        for (int i = m; i < num_rows - m; i++)
        {
            for (int j = m; j < num_cols - m; j++)
            {
                uchar max_val = 0;

                for (int k = -m; k <= m; k++)
                {
                    uchar pixel = src.at<uchar>(i + k, j);
                    if (pixel > max_val)
                        max_val = pixel;
                }
                for (int l = -m; l <= m; l++)
                {
                    uchar pixel = src.at<uchar>(i, j + l);
                    if (pixel > max_val)
                        max_val = pixel;
                }

                dst.at<uchar>(i, j) = max_val;
            }
        }
    }
    else
    {
        std::cout << "Connectedness can only be 4 or 8" << std::endl;
        return -1;
    }
    return 0;
}

cv::Vec3b getColorForRegion(cv::Point2d centroid, std::map<int, RegionInfo>& prevRegions) {

    for (const auto& reg : prevRegions) {
        cv::Point2d prevCentroid = reg.second.centroid;
        double distance = cv::norm(centroid - prevCentroid);

        if (distance < 50) {
            return reg.second.color;
        }
    }

    return cv::Vec3b(rand() % 256, rand() % 256, rand() % 256);
}

cv::Mat segmentObjects(cv::Mat &src, cv::Mat &dst, int minRegionSize, std::map<int, RegionInfo>& prevRegions) {

    cv::Mat labels, stats, centroids;
    int nLabels = cv::connectedComponentsWithStats(src, labels, stats, centroids, 8, CV_32S);

    dst = cv::Mat::zeros(src.size(), CV_8UC3);
    std::map<int, RegionInfo> currentRegions;

    for (int i = 1; i < nLabels; i++) {
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        cv::Point2d centroid(centroids.at<double>(i, 0), centroids.at<double>(i, 1));

        if (area > minRegionSize) {
            cv::Vec3b color = getColorForRegion(centroid, prevRegions);
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

    prevRegions = std::move(currentRegions);
    return labels;
}

cv::Moments computeFeatures(cv::Mat &src, const cv::Mat &labels, int label, const cv::Point2d &centroid, const cv::Vec3b &color) {
    cv::Mat mask = cv::Mat::zeros(labels.size(), CV_8U);
    for (int y = 0; y < labels.rows; ++y) {
        for (int x = 0; x < labels.cols; ++x) {
            if (labels.at<int>(y, x) == label) {
                mask.at<uchar>(y, x) = 255;
            }
        }
    }

    cv::Moments m = cv::moments(mask, true);
    double angle = 0.5 * std::atan2(2 * m.mu11, m.mu20 - m.mu02);

    std::vector<cv::Point> points;
    cv::findNonZero(mask, points);
    cv::RotatedRect rotRect = cv::minAreaRect(points);

    cv::Point2f rectPoints[4];
    rotRect.points(rectPoints);
    for (int j = 0; j < 4; j++) {
        cv::line(src, rectPoints[j], rectPoints[(j + 1) % 4], cv::Scalar(color), 2);
    }

    cv::Point center = rotRect.center;
    cv::Point endpoint(center.x + cos(angle) * 100, center.y + sin(angle) * 100);
    cv::line(src, center, endpoint, cv::Scalar(color), 2);
    return m;
}