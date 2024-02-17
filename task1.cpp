#include <opencv2/opencv.hpp>

int main(int argc, char *argv[]) {
    cv::VideoCapture *capdev;

    // Open the video device
    capdev = new cv::VideoCapture(1); // Change the parameter to the appropriate device index
    if (!capdev->isOpened()) {
        printf("Unable to open video device\n");
        return (-1);
    }

    cv::namedWindow("Original Video", 1); // Window for original video
    cv::namedWindow("Thresholded Video", 1); // Window for thresholded video
    cv::namedWindow("Cleaned Thresholded Video", 1); // Window for cleaned thresholded video

    cv::Mat frame, frame_gray, frame_thresholded, frame_cleaned;

    for (;;) {
        *capdev >> frame; // Get a new frame from the camera

        if (frame.empty()) {
            printf("Frame is empty\n");
            break;
        }

        // Preprocess the frame (optional)
        cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(frame_gray, frame_gray, cv::Size(5, 5), 0);

        // Thresholding
        cv::threshold(frame_gray, frame_thresholded, 100, 255, cv::THRESH_BINARY);

        // Morphological filtering (cleaning up)
        // we use the morphological closing operation (cv::MORPH_CLOSE) to 
        // fill in small holes and smooth out irregularities in the binary image.
        // We define a kernel (cv::Mat) to be used for the morphological operation. 
        // Here, we use a rectangular kernel of size 5x5

        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
        cv::morphologyEx(frame_thresholded, frame_cleaned, cv::MORPH_CLOSE, kernel);

        // Display original, thresholded, and cleaned thresholded video
        cv::imshow("Original Video", frame);
        cv::imshow("Thresholded Video", frame_thresholded);
        cv::imshow("Cleaned Thresholded Video", frame_cleaned);

        // Check for key press
        char key = cv::waitKey(10);
        if (key == 'q') {
            break; // Quit if 'q' is pressed
        }
    }

    delete capdev;
    return 0;
}
