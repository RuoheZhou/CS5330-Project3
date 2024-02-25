
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


int read_image_data_csv(char *filename, std::vector<char *> &filenames, std::vector<std::vector<float>> &data, int echo_file)
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
    filenames.push_back(fname);
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
    for (int i = 0; i < data.size(); i++)
    {
      float temp = 0;
      for (int j = 0; j < data[i].size(); j++)
      {
        // calculating SSD
        float diff = data[i][j] - targetVector[j];
        temp += diff * diff;
      }
      // taking square root
      float dist = std::sqrt(temp);

      std::string current_label = labels[i];
      image_ranks.push_back(std::make_pair(dist, current_label));
    }

    // sorting the vector pair in ascending order of the float values
    sort(image_ranks.begin(), image_ranks.end());

    std::cout<<image_ranks[1].second<<std::endl;
    std::cout<<image_ranks[2].second<<std::endl;
    std::cout<<image_ranks[3].second<<std::endl;
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
    cv::VideoCapture cap("/home/ronak/Downloads/objects.mp4");
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
        cv::resize(frame, frame, cv::Size(600, 480));
        thresholding(frame, thresholded, 100);
        dilation(thresholded, dilated, 5, 8);
        erosion(dilated, eroded, 5, 4);
        cv::Mat labels = segmentObjects(eroded, segmented, 500, prevRegions);

        char key = static_cast<char>(cv::waitKey(1));
        if (key == 'N' || key == 'n')
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