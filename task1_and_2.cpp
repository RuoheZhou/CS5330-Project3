/**
 * @file task1_and_2.cpp
 * @author Ronak Bhanushali and Ruohe Zhou
 * @brief Task 1 and 2
 * @date 2024-02-26
 * 
 */
#include <opencv2/opencv.hpp>
#include "filters.hpp"

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

    cv::Mat frame,thresholded_frame,dilated_img,eroded_img;

    for (;;) {
        *capdev >> frame; // Get a new frame from the camera
        if (frame.empty()) {
            printf("Frame is empty\n");
            break;
        }

        // Preprocess the frame (optional)

        // Thresholding
        thresholding(frame,thresholded_frame,120);
        dilation(thresholded_frame,dilated_img,5,8);
        erosion(dilated_img,eroded_img,5,4);
        
        cv::imshow("Original Video", frame);
        cv::imshow("Thresholded Video", thresholded_frame);
        cv::imshow("Cleaned Video", eroded_img);

        // Check for key press
        char key = cv::waitKey(10);
        if (key == 'q') {
            break; // Quit if 'q' is pressed
        }
    }
    return 0;
}
