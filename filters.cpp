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
