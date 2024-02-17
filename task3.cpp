#include <opencv2/opencv.hpp>
#include "filters.hpp"


int thresholding(cv::Mat & src, cv::Mat & dst, int threshold)
{
    int num_rows = src.rows;
    int num_cols = src.cols;
    cv::Mat grayscale_img,dilated_img,eroded_img;
    cv::cvtColor(src, grayscale_img, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(grayscale_img, grayscale_img, cv::Size(5, 5), 0);

    dilation(grayscale_img,dilated_img,5,8);
    erosion(dilated_img,eroded_img,5,4);
    
    cv::Mat temp = cv::Mat::zeros(src.size(), CV_8UC3); // Create a grayscale image

    for (int i = 0; i < num_rows; i++)
    {
        for (int j = 0; j < num_cols; j++)
        {
            uchar pixel_value = grayscale_img.at<uchar>(i, j);
            if (pixel_value > threshold) {
                temp.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0);
            }
            else {
                temp.at<cv::Vec3b>(i, j) = src.at<cv::Vec3b>(i, j);
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

int main(int argc, char *argv[]) {
    cv::VideoCapture *capdev;

    // Open the video device
    capdev = new cv::VideoCapture(0); // Change the parameter to the appropriate device index
    if (!capdev->isOpened()) {
        printf("Unable to open video device\n");
        return (-1);
    }

    cv::namedWindow("Original Video", 1); // Window for original video
    cv::namedWindow("Thresholded Video", 1); // Window for thresholded video

    cv::Mat frame,thresholded_frame,dilated_img;

    for (;;) {
        *capdev >> frame; // Get a new frame from the camera

        if (frame.empty()) {
            printf("Frame is empty\n");
            break;
        }

        // Preprocess the frame (optional)

        // Thresholding
        thresholding(frame,thresholded_frame,120);
        
        cv::imshow("Original Video", frame);
        cv::imshow("Thresholded Video", thresholded_frame);
        // cv::imshow("Cleaned Thresholded Video", frame_cleaned);

        // // Create a color map for the regions
        // cv::Mat region_map(frame_cleaned.size(), CV_8UC3, cv::Scalar(0, 0, 0));
        // std::vector<cv::Vec3b> colors;
        // for (int i = 0; i < num_regions; ++i) {
        //     colors.push_back(cv::Vec3b(rand() % 256, rand() % 256, rand() % 256)); // Generate random colors
        // }

        // // Draw regions on the region map
        // for (int y = 0; y < labels.rows; ++y) {
        //     for (int x = 0; x < labels.cols; ++x) {
        //         int label = labels.at<int>(y, x);
        //         if (label > 0) {
        //             region_map.at<cv::Vec3b>(y, x) = colors[label];
        //         }
        //     }
        // }

        // cv::imshow("Regions", region_map);

        // Check for key press
        char key = cv::waitKey(17);
        if (key == 'q') {
            break; // Quit if 'q' is pressed
        }
    }

    // delete capdev;
    // cv::Mat input_image = cv::imread("/home/ronak/Downloads/j.png");
    // cv::Mat grayscale_input;
    // cv::cvtColor(input_image,grayscale_input, cv::COLOR_BGR2GRAY);
    // cv::Mat eroded_img;
    // cv::Mat dilated_img;
    // erosion(grayscale_input, eroded_img, 3);
    // dilation(grayscale_input, dilated_img, 3);
    // cv::imshow("Erosion",eroded_img);
    // cv::imshow("Dilation",dilated_img);
    // cv::imshow("Original Image",grayscale_input);
    // cv::waitKey(0);
    return 0;
}
