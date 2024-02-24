#include <opencv2/opencv.hpp>
#include "faceDetect.cpp"
#include "faceDetect_greybg.cpp"

int sobelX3x3(cv::Mat &src, cv::Mat &dst) {
    dst.create(src.size(), CV_16SC3);

    for (int y = 1; y < src.rows - 1; ++y) {
        for (int x = 1; x < src.cols - 1; ++x) {
            for (int c = 0; c < 3; ++c) {
                int sobelX = src.ptr<cv::Vec3b>(y)[x + 1][c] - src.ptr<cv::Vec3b>(y)[x - 1][c];
                dst.ptr<cv::Vec3s>(y)[x][c] = static_cast<short>(sobelX);
            }
        }
    }

    return 0;  
}
//Sample Comment
int sobelY3x3(cv::Mat &src, cv::Mat &dst) {
    dst.create(src.size(), CV_16SC3);

    for (int y = 1; y < src.rows - 1; ++y) {
        for (int x = 1; x < src.cols - 1; ++x) {
            for (int c = 0; c < 3; ++c) {
                int sobelY = src.ptr<cv::Vec3b>(y + 1)[x][c] - src.ptr<cv::Vec3b>(y - 1)[x][c];
                dst.ptr<cv::Vec3s>(y)[x][c] = static_cast<short>(sobelY);
            }
        }
    }

    return 0; 
}

int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst) {

    dst.create(sx.size(), CV_8UC3);

    for (int y = 0; y < sx.rows; ++y) {
        for (int x = 0; x < sx.cols; ++x) {
            for (int c = 0; c < 3; ++c) {
                // Calculate Euclidean distance for magnitude: I = sqrt( sx*sx + sy*sy )
                float mag = std::sqrt(sx.ptr<cv::Vec3s>(y)[x][c] * sx.ptr<cv::Vec3s>(y)[x][c] +
                                      sy.ptr<cv::Vec3s>(y)[x][c] * sy.ptr<cv::Vec3s>(y)[x][c]);

                // Convert the magnitude to uchar 
                dst.ptr<cv::Vec3b>(y)[x][c] = static_cast<uchar>(mag);
            }
        }
    }

    return 0;
}



int preserveStrongColor(cv::Mat &input, cv::Mat &output) {
    cv::cvtColor(input, output, cv::COLOR_BGR2HSV); // Convert to HSV color space

    // Define a threshold for the intensity (V channel)
    int intensityThreshold = 150;

    // Iterate through each pixel
    for (int y = 0; y < output.rows; ++y) {
        for (int x = 0; x < output.cols; ++x) {
            // Check if the intensity is above the threshold
            if (output.at<cv::Vec3b>(y, x)[2] > intensityThreshold) {
                // Preserve the hue and saturation values, set the intensity to the original value
                output.at<cv::Vec3b>(y, x)[2] = input.at<cv::Vec3b>(y, x)[2];
            } else {
                // Set the pixel to grey
                output.at<cv::Vec3b>(y, x) = cv::Vec3b(128, 128, 128);
            }
        }
    }

    cv::cvtColor(output, output, cv::COLOR_HSV2BGR); // Convert back to BGR color space

    return 0;
}

int main(int argc, char *argv[]) {
    cv::VideoCapture *capdev;

    // Open the video device
    capdev = new cv::VideoCapture(1);
    if (!capdev->isOpened()) {
        printf("Unable to open video device\n");
        return (-1);
    }
    // Add a delay after opening the video device
    cv::waitKey(1000);
    // Get some properties of the image
    cv::Size refS((int)capdev->get(cv::CAP_PROP_FRAME_WIDTH),
                  (int)capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
    printf("Expected size: %d %d\n", refS.width, refS.height);

    cv::namedWindow("Video", 1); // Identifies a window
    cv::Mat frame;

    char lastKeypress = 'c'; // Default to color display
    int saveCounter = 0; 
    
    for (;;) {
        *capdev >> frame; // Get a new frame from the camera, treat as a stream
        if (frame.empty()) {
            printf("Frame is empty\n");
            break;
        }

        // Check the last keypress and modify the image accordingly
        if (lastKeypress == 'g') {
            // Convert the frame to greyscale using cvtColor
            cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
        }
        
        if (lastKeypress == 'h') {
            // Convert the frame to greyscale using cvtColor
            cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);

            // Increase contrast (adjust the contrast factor as needed)
            frame.convertTo(frame, -1, 1.5, 0);
        }

      
        if (lastKeypress == 'b') {
            for (int i = 0; i < frame.rows; i++) {
                for (int j = 0; j < frame.cols; j++) {
                    for (int c = 0; c < frame.channels(); c++) {
                        frame.at<cv::Vec3b>(i, j)[c] = cv::saturate_cast<uchar>(2 * (255 - frame.at<cv::Vec3b>(i, j)[c]));
                    }
                }
            }
        }



        if (lastKeypress == 'x') {
            // Create Mat to store Sobel X output
            cv::Mat sobelXOutput;
            // Apply Sobel X filter
            sobelX3x3(frame, sobelXOutput);
            // Visualize the Sobel X output (absolute values)
            cv::Mat sobelXAbs;
            cv::convertScaleAbs(sobelXOutput, sobelXAbs);
            cv::imshow("Sobel X", sobelXAbs);
        }
        
        if (lastKeypress == 'y') {
            // Create Mat to store Sobel Y output
            cv::Mat sobelYOutput;
            // Apply Sobel Y filter
            sobelY3x3(frame, sobelYOutput);
            // Visualize the Sobel Y output (absolute values)
            cv::Mat sobelYAbs;
            cv::convertScaleAbs(sobelYOutput, sobelYAbs);
            cv::imshow("Sobel Y", sobelYAbs);
        }

        if (lastKeypress == 'm') {
            cv::Mat sobelXOutput, sobelYOutput;

            sobelX3x3(frame, sobelXOutput);
            cv::Mat sobelXAbs;
            cv::convertScaleAbs(sobelXOutput, sobelXAbs);

            sobelY3x3(frame, sobelYOutput);
            cv::Mat sobelYAbs;
            cv::convertScaleAbs(sobelYOutput, sobelYAbs);

            cv::Mat gradientMagnitude;
            magnitude(sobelXOutput, sobelYOutput, gradientMagnitude);
            cv::imshow("Gradient Magnitude", gradientMagnitude);
        }

        if (lastKeypress == 'f') {
            cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
            // Detect faces
            std::vector<cv::Rect> faces;
            detectFaces(frame, faces);

            // Draw boxes around detected faces
            drawBoxes(frame, faces, 0, 1.0);
        }

        if (lastKeypress == 'z') {
            
            // Detect faces
            std::vector<cv::Rect> faces;
            detectFaces_greybg(frame, faces);

            // Draw boxes around detected faces
            drawBoxes_greybg(frame, faces, 0, 1.0);
        }

        if (lastKeypress == 'a') {
            preserveStrongColor(frame, frame);
        }

        if (lastKeypress == 'p') {

        // Convert the frame to a negative image
            cv::bitwise_not(frame,frame);
            // Normalize pixel values to the range [0, 255]
            cv::normalize(frame, frame, 0, 255, cv::NORM_MINMAX);
            cv::imwrite("../negative_image.jpg", frame);
        
    }



        cv::imshow("Video", frame);

        // See if there is a waiting keystroke
        char key = cv::waitKey(10);
        if (key == 'q') {
            break; // Quit if 'q' is pressed
        } else if (key == 's') {
            // Save the current frame to a file if 's' is pressed
            std::string filename = "../captured_frame_" + std::to_string(saveCounter) + ".jpg";
            cv::imwrite(filename, frame);
            printf("Image saved as: %s\n", filename.c_str());
            saveCounter++;
        } else if (key == 'g' || key == 'c' || key == 'x' || key == 'y' || key == 'm' || key == 'f' || 
        key == 'z' || key == 'a' || key == 'p' || key == 'h' | key == 'b') {
            // Update the last keypress
            lastKeypress = key;
        }
    }

    delete capdev;
    return (0);
}
