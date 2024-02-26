/**
 * @file task5.cpp
 * @author Ronak Bhanushali and Ruohe Zhou
 * @brief Task 5
 * @date 2024-02-26
 * 
 */
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <string>
#include "filters.hpp"

int append_image_data_csv(char *csv_file_name, std::string object_name, std::vector<float> &image_data, int reset_file)
{
  char buffer[256];
  char mode[8];

  FILE *fp;

  std::strcpy(mode, "a");

  if (reset_file)
  {
    std::strcpy(mode, "w");
  }

  fp = fopen(csv_file_name, mode);

  if (!fp)
  {
    // printf("Unable to open output file %s\n", argv[2] );
    std::cout << "Unable to open CSV" << std::endl;
    exit(-1);
  }

  // write the filename and the feature vector to the CSV file
  // std::cout << image_filename << std::endl;
  std::strcpy(buffer, object_name.c_str());
  std::fwrite(buffer, sizeof(char), strlen(buffer), fp);
  for (int i = 0; i < image_data.size(); i++)
  {
    char tmp[256];
    sprintf(tmp, ",%.4f", image_data[i]);
    std::fwrite(tmp, sizeof(char), strlen(tmp), fp);
  }

  std::fwrite("\n", sizeof(char), 1, fp); // EOL

  fclose(fp);
  return (0);
}

int main() {

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Unable to open video device" << std::endl;
        return -1;
    }

    cv::Mat frame, thresholded, segmented, eroded, dilated;
    std::map<int, RegionInfo> prevRegions;

        while (true) {
            cap >> frame;
            if (frame.empty()) break;

            thresholding(frame, thresholded, 100);
            dilation(thresholded, dilated, 5, 8);
            erosion(dilated, eroded, 5, 4);

            cv::Mat labels = segmentObjects(eroded, segmented, 500, prevRegions);
            int key = cv::waitKey(30);
            if (key == 'N' || key == 'n')
            {
                std::string obj_name;
                std::cout << "Enter a name/label for the moments data: ";
                std::cin >> obj_name;

                for (const auto &reg : prevRegions)
                {
                  cv::Moments m = computeFeatures(frame, labels, reg.first, reg.second.centroid, reg.second.color);

                  double huMoments[7];
                  cv::HuMoments(m, huMoments);

                  std::vector<float> input_data(huMoments, huMoments + 7);
                  append_image_data_csv("../data/features.csv", obj_name, input_data, 0);
                }
                std::cout << "DATA SAVED" << std::endl;
            }
            else if(key == 'q')
            {
              cv::destroyAllWindows();
              exit(0);
            }
            else
            {
              for (const auto &reg : prevRegions)
              {
                computeFeatures(frame, labels, reg.first, reg.second.centroid, reg.second.color);
              }
            }
            cv::imshow("Output", frame);
        }
        return 0;
    }
