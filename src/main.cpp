/**
 * @file HoughCircle_Demo.cpp
 * @brief Demo code for Hough Transform
 * @author OpenCV team
 */

#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <math.h>

// g++ -std=c++14 ex3_b.cpp -o app `pkg-config --cflags --libs opencv` && ./app

using namespace std;
using namespace cv;

// windows and trackbars name
const std::string windowName = "Hough Circle Detection Demo";
const std::string cannyThresholdTrackbarName = "Canny threshold";
const std::string accumulatorThresholdTrackbarName = "Accumulator Threshold";

// values and max values of the parameters of interests.
Mat src, src_gray;
int cannyThreshold = 131;
int accumulatorThreshold = 100;
const int maxAccumulatorThreshold = 200;
const int maxCannyThreshold = 255;

int min_threshold = 0;
int max_trackbar = 150;
int p_trackbar = 83;

// TODO: proper error handling
//       beautiful code

double angleBetweenTwoLines(const Point &vec1, const Point &vec2) {
  if (vec1 == vec2)
    return 0;
  double ang = acos(vec1.dot(vec2) / (norm(vec1) * norm(vec2)));
  return (ang * 180.0f) / M_PI;
}

void swapPoints(Vec4i &line) {
  auto px = line[0];
  auto py = line[1];
  line[0] = line[2];
  line[1] = line[3];
  line[2] = px;
  line[3] = py;
}

vector<Vec4i> merge(vector<Vec4i> &clock_lines) {
  for (size_t x = 0; x < clock_lines.size() - 1; x++) {
    Vec4i l1 = clock_lines[x];

    Point p1_1 = Point(l1[0], l1[1]);
    Point p1_2 = Point(l1[2], l1[3]);

    Point vec1 = p1_1 - p1_2;

    for (size_t y = x + 1; y < clock_lines.size(); y++) {
      Vec4i l2 = clock_lines[y];

      Point p2_1 = Point(l2[0], l2[1]);
      Point p2_2 = Point(l2[2], l2[3]);

      Point vec2 = p2_1 - p2_2;

      double ang = angleBetweenTwoLines(vec1, vec2);
      if (ang < 5.f) {
        double dist1 = norm(vec1);
        double dist2 = norm(vec2);
        if (dist1 < dist2)
          clock_lines.erase(clock_lines.begin() + x);
        else
          clock_lines.erase(clock_lines.begin() + y);
        return merge(clock_lines);
      }
    }
  }
  return clock_lines;
}

void Circular_hough(int, void *) {
  // will hold the results of the detection
  std::vector<Vec3f> circles;
  // runs the actual detection (1,30,100,30,70,80)
  HoughCircles(src_gray, circles, HOUGH_GRADIENT,
               1, //
               src_gray.rows / 8, cannyThreshold,
               accumulatorThreshold, // is the aculmulator threash
               0, 0 // change the last two parameters (min radius and max radios
                    // ) to detect large circles
               );

  // clone the colour, input image for displaying purposes
  Mat display, clock; // = src.clone();
  if (circles.size() < 1) {
    return;
  }
  // ------------------------------------------------------------------
  Point center(cvRound(circles[0][0]), cvRound(circles[0][1]));
  int radius = cvRound(circles[0][2]);
  // circle center
  // circle(display, center, 3, Scalar(0, 255, 0), -1, 8, 0);
  // circle outline
  // circle(display, center, radius, Scalar(0, 0, 255), 3, 8, 0);

  // remove background and isolate clock
  // center and radius are the results of HoughCircle
  // mask is a CV_8UC1 image with 0
  cv::Mat mask = cv::Mat::zeros(src.rows, src.cols, CV_8UC1);
  circle(mask, center, radius, Scalar(255, 255, 255), -1, 8, 0);
  src.copyTo(clock, mask); // copy values of img to dst if mask is > 0.
  // ------------------------------------------------------------------

  // cv::Mat roi( display, cv::Rect( center.x-radius, center.y-radius, radius*2,
  // radius*2 ) );
  // detect hour and minut pointers
  // pre process
  Mat src_gray, src_gray_blured, pre_processed_img;
  // Pass the RGB image to grayScale
  cvtColor(clock, src_gray, COLOR_RGB2GRAY);

  // OPTIONAL (bluring image to process only the more linear edges)
  bilateralFilter(src_gray, src_gray_blured, 25, 150, BORDER_DEFAULT);

  // Apply Canny edge detector (is requeired to first apply a edge detection
  // algoritm)
  Canny(src_gray_blured, pre_processed_img, 50, 200, 3);

  // detect
  vector<Vec4i> p_lines;
  // show binary image
  cvtColor(pre_processed_img, display, COLOR_GRAY2BGR);

  /// 2. Use Probabilistic Hough Transform
  HoughLinesP(pre_processed_img, p_lines, 1, CV_PI / 180,
              min_threshold + p_trackbar, 30, 10);

  // 1º filtering the vector lines (selecting only the lines near the center)

  // 1º ver 2 pontos da linha perto do centro and organize them (remove the rest
  // and the ones that stay are organized from center to limit of the clock
  // hand)
  float clock_radius_limit = 0.1f * ((float)radius);
  vector<Vec4i> clock_lines;

  for (size_t x = 0; x < p_lines.size(); x++) {
    Vec4i line = p_lines[x];

    Point p1 = Point(line[0], line[1]);
    Point p2 = Point(line[2], line[3]);

    Point vec1 = p1 - center;
    Point vec2 = p2 - center;

    if (norm(vec1) <= clock_radius_limit && norm(vec2) > clock_radius_limit) {
      clock_lines.push_back(line);
    }
    if (norm(vec2) <= clock_radius_limit && norm(vec1) > clock_radius_limit) {
      swapPoints(line);
      clock_lines.push_back(line);
    }
  }

  // ver angulo entre eles e depois juntar linhas com angulos parecidos
  // calculating angle between them (5 degres for exemple or less)

  // merge lines
  vector<Vec4i> merged_clock_lines = merge(clock_lines);

  /// Show the result
  cout << merged_clock_lines.size() << endl;

  Point p1_1, p1_2, p2_1, p2_2;
  if (merged_clock_lines.size() > 0) {

    Vec4i l1 = merged_clock_lines[0];

    p1_1 = Point(l1[0], l1[1]);
    p1_2 = Point(l1[2], l1[3]);
    line(display, p1_1, p1_2, Scalar(0, 0, 255), 3, LINE_AA);
  }
  if (merged_clock_lines.size() > 1) {
    Vec4i l2 = merged_clock_lines[1];

    p2_1 = Point(l2[0], l2[1]);
    p2_2 = Point(l2[2], l2[3]);

    Point vec1 = p1_1 - p1_2;
    Point vec2 = p2_1 - p2_2;

    cout << "vec1: " << vec1 << ", vec2: " << vec2
         << ",  ang: " << angleBetweenTwoLines(vec1, vec2) << endl;
    line(display, p2_1, p2_2, Scalar(0, 255, 0), 3, LINE_AA);
  }
  // shows the results
  // line(display, center, Point(center.x + radius,center.y), Scalar(0, 255, 0), 3, LINE_AA);
  imshow(windowName, display);
}

int main(int argc, char **argv) {
  // Read the image
  String imageName("../data/w.jpeg"); // by default
  if (argc > 1) {
    imageName = argv[1];
  }
  src = imread(imageName, IMREAD_COLOR);

  if (src.empty()) {
    std::cerr << "Invalid input image\n";
    return -1;
  }

  // Convert it to gray
  cvtColor(src, src_gray, COLOR_BGR2GRAY);

  // Reduce the noise so we avoid false circle detection
  GaussianBlur(src_gray, src_gray, Size(9, 9), 2, 2);

  // create the main window, and attach the trackbars
  namedWindow(windowName, WINDOW_AUTOSIZE);
  createTrackbar(cannyThresholdTrackbarName, windowName, &cannyThreshold,
                 maxCannyThreshold, Circular_hough);
  createTrackbar(accumulatorThresholdTrackbarName, windowName,
                 &accumulatorThreshold, maxAccumulatorThreshold,
                 Circular_hough);
  char thresh_label[50];
  sprintf(thresh_label, "Thres: %d + input", min_threshold);
  createTrackbar(thresh_label, windowName, &p_trackbar, max_trackbar,
                 Circular_hough);

  // those paramaters cannot be =0
  // so we must check here
  cannyThreshold = max(cannyThreshold, 1);
  accumulatorThreshold = max(accumulatorThreshold, 1);

  // initializes
  Circular_hough(0, 0);
  // get user key
  waitKey(0);

  return 0;
}