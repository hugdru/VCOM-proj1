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
  double radius;

  explicit Circle(Vec3f &circleData) {
    center.x = circleData[0];
    center.y = circleData[1];
    radius = circleData[2];
  }
};

struct Line {
  Point a; // center point
  Point b; // edge point

  Line() = default;

  explicit Line(const Point &p1, const Point &p2) {
    a.x = p1.x;
    a.y = p1.y;
    b.x = p2.x;
    b.y = p2.y;
  }
  explicit Line(Vec4i rawLine) {
    a.x = rawLine[0];
    a.y = rawLine[1];
    b.x = rawLine[2];
    b.y = rawLine[3];
  }
};

struct TimeExtracted {
  int hour;
  int minute;

  TimeExtracted() : hour(0), minute(0) {}
  explicit TimeExtracted(int h, int m) : hour(h), minute(m) {}
};

string readCommandLine(int argc, char **argv, string const &defaultImagePath);

void readImage(string &imagePath, ProgramData &programData);

void buildGui(TrackbarCallback callback, ProgramData &programData);

void clockTimeDetector(int, void *);

vector<Circle> getCircles(ProgramData &programData);

vector<Line> getPointerMergedLines(Mat &src, Mat &result,
                                   ProgramData &programData,
                                   const Circle &clockCircle);

void isolateClock(Circle &clockCircle, Mat &image, Mat &clock);

vector<Line> selectLinesCloseToCircleCenter(vector<Line> &lines,const Circle &circle,
                                            double radiusFactor);
vector<Line> naiveClockPointerLinesMerge(vector<Line> &clockLines);

TimeExtracted extractTime(const vector<Line> &mergedClockLines,
                          const Circle &circle);

ostream &operator<<(ostream &ostr, const TimeExtracted &time);
double angleBetweenTwoLines(const Point &vec1, const Point &vec2);
double clockWiseAngleBetweenTwoVectors(const Point &vec1, const Point &vec2);
int getHourFromAngleDeg(double angle);
int getMinuteFromAngleDeg(double angle);
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
  vector<Line> mergedClockLines =
      getPointerMergedLines(grayClock, display, *programData, clockCircle);
  gray2bgr(display, display);

  TimeExtracted hoursExtracted = extractTime(mergedClockLines, clockCircle);

  cout << hoursExtracted << endl;

  // show lines
  cout << "Merged clock lines size: " << mergedClockLines.size() << endl;
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

  Point limitPoint;
  limitPoint.x = clockCircle.center.x;
  limitPoint.y = clockCircle.center.y - clockCircle.radius;
  Line midNightLine = Line(clockCircle.center, limitPoint);
  line(display, midNightLine.a, midNightLine.b, Scalar(0, 255, 255), 3,
       LINE_AA);

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

vector<Line> getPointerMergedLines(Mat &src, Mat &result,
                                   ProgramData &programData,
                                   const Circle &clockCircle) {
  bilateralFilter(src, result, 25, 150, BORDER_DEFAULT);
  Canny(result, result, 50, 200, 3);

  // TODO: improve this method (play arround with values, improve method, change
  // the increase of threashhold and other things)
  vector<Line> mergedClockLines;
  int trys = 0;
  do {
    vector<Vec4i> rawLines;
    HoughLinesP(result, rawLines, 1, CV_PI / 180,
                (programData.houghLinesPThreshold + trys * 5) %
                    MAX_HOUGH_LINES_P_THRESHOLD,
                30, 10);
    vector<Line> lines;
    for (auto &rawLine : rawLines) {
      lines.push_back(Line(rawLine));
    }

    vector<Line> clockLines = selectLinesCloseToCircleCenter(lines, clockCircle, LINES_SELECTION_RADIUS_FACTOR);
    mergedClockLines = naiveClockPointerLinesMerge(clockLines);
    cout << "size: " << mergedClockLines.size() << endl;
    if (mergedClockLines.size() >= 2)
      break;
  } while ((trys++) < 5);
  return mergedClockLines;
}

void isolateClock(Circle &clockCircle, Mat &image, Mat &clock) {
  cv::Mat mask = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);
  circle(mask, clockCircle.center, clockCircle.radius, Scalar(255, 255, 255),
         -1, LINE_8, 0);
  image.copyTo(clock, mask);
}

vector<Line> selectLinesCloseToCircleCenter(vector<Line> &lines,const Circle &circle,
                                            double radiusFactor) {
  double clock_radius_limit = radiusFactor * static_cast<double>(circle.radius);
  vector<Line> clockPointerLines;

  for (auto &line : lines) {
    Point vec1 = line.a - circle.center;
    Point vec2 = line.b - circle.center;

    if (norm(vec1) <= clock_radius_limit && norm(vec2) > clock_radius_limit) {
      clockPointerLines.push_back(line);
    }
    if (norm(vec2) <= clock_radius_limit && norm(vec1) > clock_radius_limit) {
      swapPoints(line);
      clockPointerLines.push_back(line);
    }
  }
  return clockPointerLines;
}

vector<Line> naiveClockPointerLinesMerge(vector<Line> &clockLines) {
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
        return naiveClockPointerLinesMerge(clockLines);
      }
    }
  }
  return clockLines;
}

TimeExtracted extractTime(const vector<Line> &mergedClockLines,
                          const Circle &circle) {
  // we have 1 or two lines
  if (mergedClockLines.size() <= 0)
    return TimeExtracted();

  // determine mid night clock pointer to help determine the corret hour and
  // minute
  Point limitPoint;
  limitPoint.x = circle.center.x;
  limitPoint.y = circle.center.y - circle.radius;
  Line midNightLine = Line(circle.center, limitPoint);
  Point vecReference = midNightLine.b - midNightLine.a;

  // get the first clock pointer
  Line pointer_1 = mergedClockLines[0];
  Point vecPointer_1 = pointer_1.b - pointer_1.a;
  double ang_1 = clockWiseAngleBetweenTwoVectors(vecReference, vecPointer_1);

  if (mergedClockLines.size() >= 2) {
    // get the secound clock pointer
    Line pointer_2 = mergedClockLines[1];
    Point vecPointer_2 = pointer_2.b - pointer_2.a;
    double ang_2 = clockWiseAngleBetweenTwoVectors(vecReference, vecPointer_2);

    // compare sizes the bigger is the minute pointer and the other de hour
    // pointer
    TimeExtracted time;
    double size_vec_1 = norm(vecPointer_1);
    double size_vec_2 = norm(vecPointer_2);

    time.hour = getHourFromAngleDeg(size_vec_1 > size_vec_2 ? ang_2 : ang_1);
    time.minute =
        getMinuteFromAngleDeg(size_vec_1 > size_vec_2 ? ang_1 : ang_2);
    return time;
  }

  // return value of only one pointer by default (clock pointer are overlaped)
  return TimeExtracted(getHourFromAngleDeg(ang_1),
                       getMinuteFromAngleDeg(ang_1));
}

// TODO: define for hours and degree
int getHourFromAngleDeg(double angle) { return (int)((angle * 12.0) / 360.0); }

int getMinuteFromAngleDeg(double angle) {
  return (int)((angle * 60.0) / 360.0);
}

ostream &operator<<(ostream &ostr, const TimeExtracted &time) {
  ostr << "Hours: " << time.hour << ":" << time.minute;
  return ostr;
}

double clockWiseAngleBetweenTwoVectors(const Point &vec1, const Point &vec2) {
  if (vec1 == vec2) {
    return 0;
  }
  double dot =
      vec1.x * vec2.x + vec1.y * vec2.y; // dot product between vec1 and vec2
  double det = vec1.x * vec2.y - vec1.y * vec2.x; // determinant
  double ang = atan2(det, dot); // atan2(y, x) or atan2(sin, cos)
  if (ang < 0)
    ang = 2 * M_PI + ang;
  return (ang * 180.0) / M_PI;
}

// TODO: verefi is one of the norms is zero...
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
