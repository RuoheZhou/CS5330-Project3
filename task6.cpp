/**
 * @file task6.cpp
 * @author Ronak Bhanushali and Ruohe Zhou
 * @brief Task 6
 * @date 2024-02-26
 * 
 */
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <map>
#include <fstream>
#include <sstream> 
#include "filters.hpp"

int getstring(FILE *fp, char os[])
{
  int p = 0;
  int eol = 0;

  for (;;)
  {
    char ch = fgetc(fp);
    if (ch == ',')
    {
      break;
    }
    else if (ch == '\n' || ch == EOF)
    {
      eol = 1;
      break;
    }
    // printf("%c", ch ); // uncomment for debugging
    os[p] = ch;
    p++;
  }
  // printf("\n"); // uncomment for debugging
  os[p] = '\0';

  return (eol); // return true if eol
}

int getint(FILE *fp, int *v)
{
  char s[256];
  int p = 0;
  int eol = 0;

  for (;;)
  {
    char ch = fgetc(fp);
    if (ch == ',')
    {
      break;
    }
    else if (ch == '\n' || ch == EOF)
    {
      eol = 1;
      break;
    }

    s[p] = ch;
    p++;
  }
  s[p] = '\0'; // terminator
  *v = atoi(s);

  return (eol); // return true if eol
}

/*
  Utility function for reading one float value from a CSV file

  The value is stored in the v parameter

  The function returns true if it reaches the end of a line or the file
 */
int getfloat(FILE *fp, float *v)
{
  char s[256];
  int p = 0;
  int eol = 0;

  for (;;)
  {
    char ch = fgetc(fp);
    if (ch == ',')
    {
      break;
    }
    else if (ch == '\n' || ch == EOF)
    {
      eol = 1;
      break;
    }

    s[p] = ch;
    p++;
  }
  s[p] = '\0'; // terminator
  *v = atof(s);

  return (eol); // return true if eol
}

int read_image_data_csv(char *filename, std::vector<char *> &labels, std::vector<std::vector<float>> &data, int echo_file)
{
  FILE *fp;
  float fval;
  char img_file[256];

  fp = fopen(filename, "r");
  if (!fp)
  {
    printf("Unable to open feature file\n");
    return (-1);
  }

  printf("Reading %s\n", filename);
  for (;;)
  {
    std::vector<float> dvec;

    // read the filename
    if (getstring(fp, img_file))
    {
      break;
    }

    // read the whole feature file into memory
    for (;;)
    {
      // get next feature
      float eol = getfloat(fp, &fval);
      dvec.push_back(fval);
      if (eol)
        break;
    }

    data.push_back(dvec);

    char *fname = new char[strlen(img_file) + 1];
    strcpy(fname, img_file);
    labels.push_back(fname);
  }
  fclose(fp);
  printf("Finished reading CSV file\n");

  if (echo_file)
  {
    for (int i = 0; i < data.size(); i++)
    {
      for (int j = 0; j < data[i].size(); j++)
      {
        printf("%.4f  ", data[i][j]);
      }
      printf("\n");
    }
    printf("\n");
  }

  return (0);
}

std::vector<std::pair<float, std::string>> calculate_scaled_euclidean_distances(const std::vector<char *>& labels, const std::vector<std::vector<float>>& known_data, const std::vector<float>& new_value) {
    std::vector<std::pair<float, std::string>> scaled_distances;

    for (size_t i = 0; i < labels.size(); ++i) {
        const std::vector<float>& data_for_label = known_data[i];

        float mean = 0;
        float std_dev = 0;

        for (float value : data_for_label) {
            mean += value;
        }
        mean /= data_for_label.size();

        for (float value : data_for_label) {
            float diff = value - mean;
            std_dev += diff * diff;
        }
        std_dev = sqrt(std_dev / data_for_label.size());

        float scaled_diff = (new_value[i] - mean) / std_dev;

        float distance = sqrt(scaled_diff * scaled_diff);

        scaled_distances.push_back(std::make_pair(distance, labels[i]));
    }

    return scaled_distances;
}

std::vector<std::pair<float, std::string>> calculate_euclidean_distances(const std::vector<char *> &labels, const std::vector<std::vector<float>> &known_data, const std::vector<float> &new_value)
{
    std::vector<std::pair<float, std::string>> distances;

    for (size_t i = 0; i < labels.size(); ++i)
    {
        const std::vector<float> &data_for_label = known_data[i];
        size_t num_features = data_for_label.size();
        float temp = 0;

        for (int j = 0; j < num_features; j++)
        {
            // calculating SSD
            float diff = data_for_label[j] - new_value[j];
            temp += diff * diff;
        }
        // taking square root
        float distance = std::sqrt(temp);

        distances.push_back(std::make_pair(distance, labels[i]));
    }

    return distances;
}

// Function to compare the feature vector of the target image with the feature vectors in the CSV file
int compareFeatures(std::vector<float> targetVector, char* csvFileName)
{
  std::vector<char *> labels;
  std::vector<std::vector<float>> data;
  int echo = 0;
  std::vector<std::pair<float, std::string>> image_ranks; // defining a vector pair (float,string)

  // read the csv file
  int result = read_image_data_csv(csvFileName, labels, data, echo);

  if (result == 0)
  {
    image_ranks = calculate_euclidean_distances(labels,data,targetVector);
    // image_ranks = calculate_scaled_euclidean_distances(labels,data,targetVector);
    // sorting the vector pair in ascending order of the float values
    sort(image_ranks.begin(), image_ranks.end());

    std::cout<<image_ranks[0].second<<","<<image_ranks[0].first<<std::endl;
    std::cout<<image_ranks[1].second<<","<<image_ranks[1].first<<std::endl;
    std::cout<<image_ranks[2].second<<","<<image_ranks[2].first<<std::endl;
    // Free allocated memory in the filenames vector
    for (char *fname : labels)
    {
      delete[] fname;
    }
  }

  else
  {
    std::cerr << "Error reading CSV file.\n";
  }
  printf("Terminating second program\n");
  return (0);
}


int main() {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Unable to open video device" << std::endl;
        return -1;
    }

    cv::namedWindow("Segmented", cv::WINDOW_AUTOSIZE);

    // Load the feature database from the specified CSV file
    cv::Mat frame, thresholded, segmented, eroded, dilated;
    std::map<int, RegionInfo> prevRegions;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;
        thresholding(frame, thresholded, 100);
        dilation(thresholded, dilated, 5, 8);
        erosion(dilated, eroded, 5, 4);
        cv::Mat labels = segmentObjects(eroded, segmented, 500, prevRegions);

        char key = static_cast<char>(cv::waitKey(1));
        if (key == 'i')
        {
            for (const auto &reg : prevRegions)
            {
                cv::Moments m = cv::moments(labels == reg.first, true); // Recompute moments for this region
                double huMoments[7];
                cv::HuMoments(m, huMoments);

                std::vector<float> features(huMoments, huMoments + 7);

                compareFeatures(features, "../data/features.csv");
            }
        }
        else
        {
            for (const auto &reg : prevRegions)
            {
                computeFeatures(frame, labels, reg.first, reg.second.centroid, reg.second.color);
            }
        }

        if (key == 'q' || key == 27) break;
        cv::imshow("Original Video", frame);
    }

    cv::destroyAllWindows();
    return 0;
}
