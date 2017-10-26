#include <algorithm>
#include <cmath>
#include <iostream>

#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

using namespace std;
using namespace cv;

const string DEFAULT_IMAGE_PATH = "../data/w.jpeg";

const string WINDOW_NAME = "Clock Time Detection";
const string HOUGH_CIRCLES_CANNY_THRESHOLD_TRACKBAR_NAME =
    "HoughCircles Canny Threshold";
const string HOUGH_CIRCLES_ACCUMULATOR_THRESHOLD_TRACKBAR_NAME =
    "HoughCircles Accumulator Threshold";
const string HOUGH_LINES_P_TRACKBAR_NAME = "HoughLinesP Threshold";

constexpr int DEFAULT_HOUGH_CIRCLES_CANNY_THRESHOLD = 131;
constexpr int MIN_HOUGH_CIRCLES_CANNY_THRESHOLD = 1;
constexpr int MAX_HOUGH_CIRCLES_CANNY_THRESHOLD = 255;

constexpr int DEFAULT_HOUGH_CIRCLES_ACCUMULATOR_THRESHOLD = 100;
constexpr int MIN_HOUGH_CIRCLES_ACCUMULATOR_THRESHOLD = 1;
constexpr int MAX_HOUGH_CIRCLES_ACCUMULATOR_THRESHOLD = 255;

constexpr int DEFAULT_HOUGH_LINES_P_THRESHOLD = 83;
constexpr int MIN_HOUGH_LINES_P_THRESHOLD = 0;
constexpr int MAX_HOUGH_LINES_P_THRESHOLD = 155;

struct ProgramData {
  Mat image;
  int houghCirclesCannyThreshold = DEFAULT_HOUGH_CIRCLES_CANNY_THRESHOLD;
  int houghCirclesAccumulatorThreshold =
      DEFAULT_HOUGH_CIRCLES_ACCUMULATOR_THRESHOLD;
  int houghLinesPThreshold = DEFAULT_HOUGH_LINES_P_THRESHOLD;
};

string readCommandLine(int argc, char **argv, string const &defaultImagePath);
void readImage(string &imagePath, ProgramData &programData);
void buildGui(TrackbarCallback callback, ProgramData &programData);
void clockTimeDetector(int, void *);
vector<Vec4i> naiveClockHandLinesMerge(vector<Vec4i> &clock_lines);
void swapPoints(Vec4i &line);
double angleBetweenTwoLines(const Point &vec1, const Point &vec2);
void convertToGray(Mat &src, Mat &dst);
template <class T>
constexpr const T &clamp(const T &v, const T &lo, const T &hi);
void normalizeHoughCirclesCannyThreshold(int &value);
void normalizeHoughCirclesAccumulatorThreshold(int &value);
void normalizeHoughLinesPThreshold(int &value);

int main(int argc, char **argv) {
  string imagePath = readCommandLine(argc, argv, DEFAULT_IMAGE_PATH);
  ProgramData programData = ProgramData();
  readImage(imagePath, programData);
  buildGui(clockTimeDetector, programData);
  clockTimeDetector(0, &programData);
  waitKey(0);
}

void clockTimeDetector(int, void *rawprogramData) {
  ProgramData *programData = static_cast<ProgramData *>(rawprogramData);

  Mat grayImage;
  convertToGray(programData->image, grayImage);

  // Reduce the noise so we avoid false circle detection
  GaussianBlur(grayImage, grayImage, Size(9, 9), 2, 2);

  std::vector<Vec3f> circles;

  HoughCircles(grayImage, circles, HOUGH_GRADIENT,
               1,  //
               grayImage.rows / 8, programData->houghCirclesCannyThreshold,
               programData->houghCirclesAccumulatorThreshold, 0,
               0  // change the last two parameters (min radius and max radios
                  // ) to detect large circles
  );

  // clone the colour, input image for displaying purposes
  Mat display, clock;  // = src.clone();
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
  cv::Mat mask =
      cv::Mat::zeros(programData->image.rows, programData->image.cols, CV_8UC1);
  circle(mask, center, radius, Scalar(255, 255, 255), -1, 8, 0);
  programData->image.copyTo(clock,
                            mask);  // copy values of img to dst if mask is > 0.
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
              programData->houghLinesPThreshold, 30, 10);

  // 1º filtering the vector lines (selecting only the lines near the center)

  // 1º ver 2 pontos da linha perto do centro and organize them (remove the rest
  // and the ones that stay are organized from center to limit of the clock
  // hand)
  double clock_radius_limit = 0.1 * static_cast<double>(radius);
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
  vector<Vec4i> merged_clock_lines = naiveClockHandLinesMerge(clock_lines);

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
  // line(display, center, Point(center.x + radius,center.y), Scalar(0, 255, 0),
  // 3, LINE_AA);
  imshow(WINDOW_NAME, display);
}

vector<Vec4i> naiveClockHandLinesMerge(vector<Vec4i> &clock_lines) {
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
      if (ang < 5.0) {
        double dist1 = norm(vec1);
        double dist2 = norm(vec2);
        if (dist1 < dist2) {
          clock_lines.erase(clock_lines.begin() + x);
        } else {
          clock_lines.erase(clock_lines.begin() + y);
        }
        return naiveClockHandLinesMerge(clock_lines);
      }
    }
  }
  return clock_lines;
}

double angleBetweenTwoLines(const Point &vec1, const Point &vec2) {
  if (vec1 == vec2) {
    return 0;
  }
  double ang = acos(vec1.dot(vec2) / (norm(vec1) * norm(vec2)));
  return (ang * 180.0) / M_PI;
}

void swapPoints(Vec4i &line) {
  swap(line[0], line[2]);
  swap(line[1], line[3]);
}

string readCommandLine(int argc, char **argv, string const &defaultImagePath) {
  string imagePath = defaultImagePath;
  if (argc > 1) {
    imagePath = argv[1];
  }
  return imagePath;
}

void readImage(string &imagePath, ProgramData &programData) {
  programData.image = imread(imagePath, IMREAD_COLOR);

  if (programData.image.empty()) {
    throw std::invalid_argument("Invalid Input Image");
  }
}

void buildGui(TrackbarCallback callback, ProgramData &programData) {
  namedWindow(WINDOW_NAME, WINDOW_AUTOSIZE);
  createTrackbar(HOUGH_CIRCLES_CANNY_THRESHOLD_TRACKBAR_NAME, WINDOW_NAME,
                 &programData.houghCirclesCannyThreshold,
                 MAX_HOUGH_CIRCLES_CANNY_THRESHOLD, callback, &programData);
  createTrackbar(HOUGH_CIRCLES_ACCUMULATOR_THRESHOLD_TRACKBAR_NAME, WINDOW_NAME,
                 &programData.houghCirclesAccumulatorThreshold,
                 MAX_HOUGH_CIRCLES_ACCUMULATOR_THRESHOLD, callback,
                 &programData);
  createTrackbar(HOUGH_LINES_P_TRACKBAR_NAME, WINDOW_NAME,
                 &programData.houghLinesPThreshold, MAX_HOUGH_LINES_P_THRESHOLD,
                 callback, &programData);
}

void normalizeHoughCirclesCannyThreshold(int &value) {
  clamp(value, MIN_HOUGH_CIRCLES_CANNY_THRESHOLD,
        MAX_HOUGH_CIRCLES_CANNY_THRESHOLD);
}

void normalizeHoughCirclesAccumulatorThreshold(int &value) {
  clamp(value, MIN_HOUGH_CIRCLES_ACCUMULATOR_THRESHOLD,
        MAX_HOUGH_CIRCLES_ACCUMULATOR_THRESHOLD);
}

void normalizeHoughLinesPThreshold(int &value) {
  clamp(value, MIN_HOUGH_LINES_P_THRESHOLD, MAX_HOUGH_LINES_P_THRESHOLD);
}

template <class T>
constexpr const T &clamp(const T &v, const T &lo, const T &hi) {
  return std::max(lo, std::min(v, hi));
}

void convertToGray(Mat &src, Mat &dst) { cvtColor(src, dst, COLOR_BGR2GRAY); }
