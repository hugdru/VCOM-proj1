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

constexpr double LINES_SELECTION_RADIUS_FACTOR = 0.1;

struct ProgramData {
  Mat image;
  Mat grayImage;
  int houghCirclesCannyThreshold = DEFAULT_HOUGH_CIRCLES_CANNY_THRESHOLD;
  int houghCirclesAccumulatorThreshold =
      DEFAULT_HOUGH_CIRCLES_ACCUMULATOR_THRESHOLD;
  int houghLinesPThreshold = DEFAULT_HOUGH_LINES_P_THRESHOLD;
};

struct Circle {
  Point center;
  int radius;

  explicit Circle(Vec3f &circleData) {
    center.x = cvRound(circleData[0]);
    center.y = cvRound(circleData[1]);
    radius = cvRound(circleData[2]);
  }
};

struct Line {
  Point a;
  Point b;

  Line() = default;

  explicit Line(Vec4i rawLine) {
    a.x = rawLine[0];
    a.y = rawLine[1];
    b.x = rawLine[2];
    b.y = rawLine[3];
  }
};

string readCommandLine(int argc, char **argv, string const &defaultImagePath);
void readImage(string &imagePath, ProgramData &programData);
void buildGui(TrackbarCallback callback, ProgramData &programData);
void clockTimeDetector(int, void *);
vector<Circle> getCircles(ProgramData &programData);
vector<Line> getLines(Mat &src, Mat &result, ProgramData &programData);
void isolateClock(Circle &clockCircle, Mat &image, Mat &clock);
vector<Line> selectLinesCloseToCircleCenter(vector<Line> &lines, Circle &circle,
                                            double radiusFactor);
vector<Line> naiveClockHandLinesMerge(vector<Line> &clockLines);

double angleBetweenTwoLines(const Point &vec1, const Point &vec2);
void bgr2gray(Mat &src, Mat &dst);
void gray2bgr(Mat &src, Mat &dst);
void normalizeHoughCirclesCannyThreshold(int &value);
void normalizeHoughCirclesAccumulatorThreshold(int &value);
void normalizeHoughLinesPThreshold(int &value);
void swapPoints(Line &line);
template <class T>
constexpr const T &clamp(const T &v, const T &lo, const T &hi);

int main(int argc, char **argv) {
  string imagePath = readCommandLine(argc, argv, DEFAULT_IMAGE_PATH);
  ProgramData programData = ProgramData();
  readImage(imagePath, programData);
  buildGui(clockTimeDetector, programData);
  clockTimeDetector(0, &programData);
  waitKey(0);
}

void clockTimeDetector(int, void *rawProgramData) {
  auto *programData = static_cast<ProgramData *>(rawProgramData);

  vector<Circle> circles = getCircles(*programData);
  if (circles.empty()) {
    return;
  }

  Circle clockCircle = circles[0];

  Mat clock;
  isolateClock(clockCircle, programData->image, clock);
  Mat grayClock;
  bgr2gray(clock, grayClock);

  Mat display;
  vector<Line> lines = getLines(grayClock, display, *programData);
  gray2bgr(display, display);

  vector<Line> clockLines = selectLinesCloseToCircleCenter(
      lines, clockCircle, LINES_SELECTION_RADIUS_FACTOR);

  vector<Line> mergedClockLines = naiveClockHandLinesMerge(clockLines);

  cout << mergedClockLines.size() << endl;

  Line line0;
  Line line1;
  if (!mergedClockLines.empty()) {
    line0 = mergedClockLines[0];
    line(display, line0.a, line0.b, Scalar(0, 0, 255), 3, LINE_AA);
  }
  if (mergedClockLines.size() > 1) {
    line1 = mergedClockLines[1];

    Point vec1 = line0.a - line0.b;
    Point vec2 = line1.a - line1.b;

    cout << "vec1: " << vec1 << ", vec2: " << vec2
         << ",  ang: " << angleBetweenTwoLines(vec1, vec2) << endl;
    line(display, line1.a, line1.b, Scalar(0, 255, 0), 3, LINE_AA);
  }

  imshow(WINDOW_NAME, display);
}

vector<Circle> getCircles(ProgramData &programData) {
  Mat blurredImage;
  GaussianBlur(programData.grayImage, blurredImage, Size(9, 9), 2, 2);

  std::vector<Vec3f> raw_circles;

  HoughCircles(blurredImage, raw_circles, HOUGH_GRADIENT, 1,
               blurredImage.rows / 8, programData.houghCirclesCannyThreshold,
               programData.houghCirclesAccumulatorThreshold, 0, 0);

  vector<Circle> circles;
  for (auto &raw_circle : raw_circles) {
    circles.push_back(Circle(raw_circle));
  }

  return circles;
}

vector<Line> getLines(Mat &src, Mat &result, ProgramData &programData) {
  bilateralFilter(src, result, 25, 150, BORDER_DEFAULT);
  Canny(result, result, 50, 200, 3);
  vector<Vec4i> rawLines;
  HoughLinesP(result, rawLines, 1, CV_PI / 180,
              programData.houghLinesPThreshold, 30, 10);
  vector<Line> lines;
  for (auto &rawLine : rawLines) {
    lines.push_back(Line(rawLine));
  }
  return lines;
}

void isolateClock(Circle &clockCircle, Mat &image, Mat &clock) {
  cv::Mat mask = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);
  circle(mask, clockCircle.center, clockCircle.radius, Scalar(255, 255, 255),
         -1, LINE_8, 0);
  image.copyTo(clock, mask);
}

vector<Line> selectLinesCloseToCircleCenter(vector<Line> &lines, Circle &circle,
                                            double radiusFactor) {
  double clock_radius_limit = radiusFactor * static_cast<double>(circle.radius);
  vector<Line> clockHandLines;

  for (auto &line : lines) {
    Point vec1 = line.a - circle.center;
    Point vec2 = line.b - circle.center;

    if (norm(vec1) <= clock_radius_limit && norm(vec2) > clock_radius_limit) {
      clockHandLines.push_back(line);
    }
    if (norm(vec2) <= clock_radius_limit && norm(vec1) > clock_radius_limit) {
      swapPoints(line);
      clockHandLines.push_back(line);
    }
  }

  return clockHandLines;
}

vector<Line> naiveClockHandLinesMerge(vector<Line> &clockLines) {
  for (size_t x = 0; x < clockLines.size() - 1; x++) {
    Line l1 = clockLines[x];

    Point vec1 = l1.a - l1.b;

    for (size_t y = x + 1; y < clockLines.size(); y++) {
      Line l2 = clockLines[y];

      Point vec2 = l2.a - l2.b;

      double ang = angleBetweenTwoLines(vec1, vec2);
      if (ang < 5.0) {
        double dist1 = norm(vec1);
        double dist2 = norm(vec2);
        if (dist1 < dist2) {
          clockLines.erase(clockLines.begin() + x);
        } else {
          clockLines.erase(clockLines.begin() + y);
        }
        return naiveClockHandLinesMerge(clockLines);
      }
    }
  }
  return clockLines;
}

double angleBetweenTwoLines(const Point &vec1, const Point &vec2) {
  if (vec1 == vec2) {
    return 0;
  }
  double ang = acos(vec1.dot(vec2) / (norm(vec1) * norm(vec2)));
  return (ang * 180.0) / M_PI;
}

void swapPoints(Line &line) { swap(line.a, line.b); }

string readCommandLine(int argc, char **argv, string const &defaultImagePath) {
  string imagePath = defaultImagePath;
  if (argc > 1) {
    imagePath = argv[1];
  }
  return imagePath;
}

void readImage(string &imagePath, ProgramData &programData) {
  programData.image = imread(imagePath, IMREAD_COLOR);
  bgr2gray(programData.image, programData.grayImage);

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

void bgr2gray(Mat &src, Mat &dst) { cvtColor(src, dst, COLOR_BGR2GRAY); }

void gray2bgr(Mat &src, Mat &dst) { cvtColor(src, dst, COLOR_GRAY2BGR); }
