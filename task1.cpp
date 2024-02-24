#include <opencv2/opencv.hpp>
#include <iostream>

int thresholding(cv::Mat & src, cv::Mat & dst, int threshold)
{
    int num_rows = src.rows;
    int num_cols = src.cols;
    cv::Mat grayscale_img;
    cv::cvtColor(src, grayscale_img, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(grayscale_img, grayscale_img, cv::Size(5, 5), 0);
 
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

int main() {
    cv::VideoCapture cap(1); // Open default camera
    if (!cap.isOpened()) {
        std::cerr << "Error: Unable to open video device" << std::endl;
        return -1;
    }

    cv::namedWindow("Original Video", cv::WINDOW_NORMAL);
    cv::namedWindow("Background Removed", cv::WINDOW_NORMAL);

    cv::Mat frame, background_removed;

    for (;;) {
        cap >> frame; // Capture frame

        if (frame.empty()) {
            std::cerr << "Error: Frame is empty" << std::endl;
            break;
        }

        // Convert frame to grayscale
        cv::Mat gray;
        // cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        // // Apply GaussianBlur to smooth the image and reduce noise
        // cv::GaussianBlur(gray, gray, cv::Size(5, 5), 0);

        cv::Mat thresholded;
        cv::threshold(gray, thresholded, 100, 255, cv::THRESH_BINARY_INV);
        
        // Call the thresholding function
        thresholding(frame, background_removed, 100);

        // Display original video and background-removed video
        cv::imshow("Original Video", frame);
        cv::imshow("Background Removed", background_removed);

        char key = cv::waitKey(10);
        if (key == 'q') {
            break; // Quit if 'q' is pressed
        }
    }

    return 0;
}
