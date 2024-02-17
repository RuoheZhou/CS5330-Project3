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
    cv::namedWindow("Regions", 1); // Window for regions found

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
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
        cv::morphologyEx(frame_thresholded, frame_cleaned, cv::MORPH_CLOSE, kernel);

        // Connected components analysis
        cv::Mat labels, stats, centroids;
        int num_regions = cv::connectedComponentsWithStats(frame_cleaned, labels, stats, centroids);

        // Display original, thresholded, and cleaned thresholded video
        cv::imshow("Original Video", frame);
        cv::imshow("Thresholded Video", frame_thresholded);
        cv::imshow("Cleaned Thresholded Video", frame_cleaned);

        // Create a color map for the regions
        cv::Mat region_map(frame_cleaned.size(), CV_8UC3, cv::Scalar(0, 0, 0));
        std::vector<cv::Vec3b> colors;
        for (int i = 0; i < num_regions; ++i) {
            colors.push_back(cv::Vec3b(rand() % 256, rand() % 256, rand() % 256)); // Generate random colors
        }

        // Draw regions on the region map
        for (int y = 0; y < labels.rows; ++y) {
            for (int x = 0; x < labels.cols; ++x) {
                int label = labels.at<int>(y, x);
                if (label > 0) {
                    region_map.at<cv::Vec3b>(y, x) = colors[label];
                }
            }
        }

        cv::imshow("Regions", region_map);

        // Check for key press
        char key = cv::waitKey(10);
        if (key == 'q') {
            break; // Quit if 'q' is pressed
        }
    }

    delete capdev;
    return 0;
}
