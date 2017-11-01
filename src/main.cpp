#define _USE_MATH_DEFINES
#include <algorithm>
#include <cmath>
#include <iostream>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

// TODO : m.realease() in all Mat objects
const string DEFAULT_IMAGE_PATH = "../data/clock3.JPG";

const string WINDOW_NAME = "Clock Time Detection";
const string HOUGH_CIRCLES_CANNY_THRESHOLD_TRACKBAR_NAME =
    "HoughCircles Canny Threshold";
const string HOUGH_CIRCLES_ACCUMULATOR_THRESHOLD_TRACKBAR_NAME =
    "HoughCircles Accumulator Threshold";
const string HOUGH_LINES_P_TRACKBAR_NAME = "HoughLinesP Threshold";

const string BILATERAL_SIGMA_COLOR_TRACKBAR_NAME = "Bilateral Sigma Color";

const string BILATERAL_SIGMA_SPACE_TRACKBAR_NAME = "Bilateral Sigma Space";

const string CANNY_THRESHOLD1_TRACKBAR_NAME = "Canny Threshold 1";

const string CANNY_THRESHOLD2_TRACKBAR_NAME = "Canny Threshold 2";

const string CANNY_APERTURE_SIZE_TRACKBAR_NAME = "Canny Aperture Size";

const string DOUBLE_EQUALITY_INTERVAL_RADIUS_PERCENTAGE_TRACKBAR_NAME = "Double Equality Interval Radius Percentage";

constexpr int DEFAULT_HOUGH_CIRCLES_CANNY_THRESHOLD = 60;
constexpr int MIN_HOUGH_CIRCLES_CANNY_THRESHOLD = 1;
constexpr int MAX_HOUGH_CIRCLES_CANNY_THRESHOLD = 255;

constexpr int DEFAULT_HOUGH_CIRCLES_ACCUMULATOR_THRESHOLD = 100;
constexpr int MIN_HOUGH_CIRCLES_ACCUMULATOR_THRESHOLD = 1;
constexpr int MAX_HOUGH_CIRCLES_ACCUMULATOR_THRESHOLD = 255;

constexpr int DEFAULT_HOUGH_LINES_P_THRESHOLD = 83;
constexpr int MIN_HOUGH_LINES_P_THRESHOLD = 0;
constexpr int MAX_HOUGH_LINES_P_THRESHOLD = 155;

constexpr int MAX_BILATERAL_SIGMA = 300;
constexpr int DEFAULT_BILATERAL_SIGMA_COLOR = 25;
constexpr int DEFAULT_BILATERAL_SIGMA_SPACE = 50;

constexpr int MAX_CANNY_TRESHOLD = 255;
constexpr int DEFAULT_CANNY_THRESHOLD1 = 50;
constexpr int DEFAULT_CANNY_THRESHOLD2 = DEFAULT_CANNY_THRESHOLD1 * 4;
constexpr int MAX_CANNY_APERTURE_SIZE = 7;
constexpr int DEFAULT_CANNY_APERTURE_SIZE = 3;

constexpr int DEFAULT_LINES_MERGE_ANGLE = 5;

constexpr double DEFAULT_LINES_SELECTION_RADIUS_FACTOR = 0.2;

constexpr int MAX_DOUBLE_EQUALITY_INTERVAL_RADIUS_PERCENTAGE = 100;
constexpr int DEFAULT_DOUBLE_EQUALITY_INTERVAL_RADIUS_PERCENTAGE = 15;

struct ProgramData {
  Mat origImg;
  Mat grayImage;
  Mat imgCropped;
  Mat grayImageCropped;
  int houghCirclesCannyThreshold = DEFAULT_HOUGH_CIRCLES_CANNY_THRESHOLD;
  int houghCirclesAccumulatorThreshold =
      DEFAULT_HOUGH_CIRCLES_ACCUMULATOR_THRESHOLD;
  int houghLinesPThreshold = DEFAULT_HOUGH_LINES_P_THRESHOLD;
  int bilateralSigmaColor = DEFAULT_BILATERAL_SIGMA_COLOR;
  int bilateralSigmaSpace = DEFAULT_BILATERAL_SIGMA_SPACE;
  int cannyThreshold1 = DEFAULT_CANNY_THRESHOLD1;
  int cannyThreshold2 = DEFAULT_CANNY_THRESHOLD2;
  int cannyApertureSize = DEFAULT_CANNY_APERTURE_SIZE;
  int doubleEqualityIntervalRadiusPercentage = DEFAULT_DOUBLE_EQUALITY_INTERVAL_RADIUS_PERCENTAGE;
};

struct Circle {
  Point2d center;
  double radius;

  explicit Circle(Vec3f &circleData) {
    center.x = circleData[0];
    center.y = circleData[1];
    radius = circleData[2];
  }
};

struct Line {
  Point2d a;  // center point
  Point2d b;  // edge point

  Line() = default;

  explicit Line(const Point2d &p1, const Point2d &p2) {
    a.x = p1.x;
    a.y = p1.y;
    b.x = p2.x;
    b.y = p2.y;
  }
  explicit Line(Vec4d rawLine) {
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

enum class SegmentsType {
  COLLINEAR_OVERLAPPING = 0,
  COLLINEAR_DISJOINT,
  PARALLEL,
  SEGMENTS_INTERSECTING,
  SEGMENTS_NOT_INTERSECTING
};

string readCommandLine(int argc, char **argv, string const &defaultImagePath);

void readImage(string &imagePath, ProgramData &programData);

void buildGui(TrackbarCallback callback, ProgramData &programData);

void createWindow(string windowName, int rows, int cols);

void imageShow(string windowName, Mat &image);

void clockTimeDetector(int, void *);

vector<Circle> getCircles(ProgramData &programData);

vector<Line> getPointerLines(Mat &result, ProgramData &programData,
                             const Circle &clockCircle);

void isolateClock(Circle &clockCircle, Mat &image, Mat &clock);

vector<Line> selectLinesCloseToCircleCenter(vector<Line> &lines,
                                            const Circle &circle,
                                            double radiusFactor);
vector<Line> clockPointerLinesMerge(vector<Line> &clockLines, int linesMergeAngle, Circle &clockCircle);

TimeExtracted extractTime(const vector<Line> &mergedClockLines,
                          const Circle &circle);

ostream &operator<<(ostream &ostr, const TimeExtracted &time);
double angleBetweenTwoLines(const Point2d &vec1, const Point2d &vec2, bool toDegree = true);
// https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect
SegmentsType segmentsAnalysis(Line &line1, Line &line2,
                              Point2d &intersectingPoint, double doubleEqualityInterval);
bool intervalsIntersect(double t0, double t1, double x0, double x1);
double crossProduct2d(Point2d &vec1, Point2d &vec2);
bool doubleIsZero(double value, double interval);
bool doubleEquality(double value, double reference, double interval);
Point2d calcLineVec(Line &line);
void calcPointDisplacement(Point2d &start, Point2d &displacementVector,
                           double displacementFactor, Point2d &displacedPoint);
double clockWiseAngleBetweenTwoVectors(const Point2d &vec1,
                                       const Point2d &vec2);
int getHourFromAngleDeg(double angle);
int getMinuteFromAngleDeg(double angle);
void bgr2gray(Mat &src, Mat &dst);
void gray2bgr(Mat &src, Mat &dst);
Scalar getDistinctColor(size_t index, size_t numberOfDistinctColors);
void normalizeHoughCirclesCannyThreshold(int &value);
void normalizeHoughCirclesAccumulatorThreshold(int &value);
void normalizeHoughLinesPThreshold(int &value);
void swapPoints(Line &line);
double medianHist(Mat grayImage);
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

  isolateClock(clockCircle, programData->origImg, programData->imgCropped);
  bgr2gray(programData->imgCropped, programData->grayImageCropped);

  Mat display;
  vector<Line> mergedClockLines =
      getPointerLines(display, *programData, clockCircle);
  gray2bgr(display, display);

  TimeExtracted hoursExtracted = extractTime(mergedClockLines, clockCircle);

  cout << endl <<  hoursExtracted << endl;

  cout << endl << "nergedClockLines.size() = " << mergedClockLines.size() << endl;
  for (size_t i = 0; i < mergedClockLines.size(); ++i) {
    auto &mergedClockLine = mergedClockLines[i];
    Scalar lineColor = getDistinctColor(i, mergedClockLines.size());
    cout << "line" << i << " - " << lineColor << endl;
    line(display, mergedClockLine.a, mergedClockLine.b, lineColor, 3, LINE_AA);
  }

  Point2d limitPoint;
  limitPoint.x = clockCircle.center.x;
  limitPoint.y = clockCircle.center.y - clockCircle.radius;
  Line midNightLine = Line(clockCircle.center, limitPoint);
  line(display, midNightLine.a, midNightLine.b, Scalar(0, 255, 255), 3,
       LINE_AA);

  imshow(WINDOW_NAME, display);

  if (mergedClockLines.size() == 2) {
    Point2d vec0 = calcLineVec(mergedClockLines[0]);
    Point2d vec1 = calcLineVec(mergedClockLines[1]);
    cout << "vec0: " << vec0 << ", vec1: " << vec1
         << ",  ang: " << angleBetweenTwoLines(vec0, vec1) << endl;
  }
}

vector<Circle> getCircles(ProgramData &programData) {
	
	Mat blurredImage;
	GaussianBlur(programData.grayImage, blurredImage, Size(9, 9), 2, 2);

	std::vector<Vec3f> raw_circles;

	Mat canny_output;
	int thresh = 200;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	/// Detect edges using canny
	Canny(programData.grayImage, canny_output, thresh, thresh * 2, 3);
	
	/// Find contours
	findContours(canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	/// Draw contours
	Mat drawing = Mat::zeros(programData.grayImage.size(), CV_8UC3);

	RNG rng(12345);
	vector<Point> approx;
	cout << "con " << contours.size() << endl;

	for (int i = 0; i< contours.size(); i++)
	{
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));

		approxPolyDP(contours[i], approx, 0.01*arcLength(contours[i], true), true);
		//drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, Point());
		if (approx.size() > 15) {
			drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, Point());
		}
	}

	/// Show in a window
	namedWindow("Contours", CV_WINDOW_AUTOSIZE);
	imshow("Contours", drawing);

	bgr2gray(drawing, drawing);
	namedWindow("Contours2", CV_WINDOW_AUTOSIZE);
	imshow("Contours2", drawing);

	HoughCircles(drawing, raw_circles, HOUGH_GRADIENT, 2,
		blurredImage.rows / 4, programData.houghCirclesCannyThreshold,
		programData.houghCirclesAccumulatorThreshold, 0, 0);

	vector<Circle> circles;
	for (auto &raw_circle : raw_circles) {
		circles.push_back(Circle(raw_circle));
	}
	cout << "circles " << circles.size() << endl;
	return circles;
}

vector<Line> getPointerLines(Mat &result, ProgramData &programData,
                             const Circle &clockCircle) {
  imageShow("before bilateral", programData.grayImageCropped);

  bilateralFilter(programData.grayImageCropped, result,
                  programData.bilateralSigmaColor,
                  programData.bilateralSigmaSpace, BORDER_DEFAULT);

  imageShow("after bilateral", result);

  Canny(result, result, programData.cannyThreshold1,
        programData.cannyThreshold2, programData.cannyApertureSize);

  imageShow("after canny", result);

  // TODO: improve this method (play around with values, improve method, change
  // the increase of threshold and other things
  vector<Vec4d> rawLines;
  vector<Line> mergedClockLines;
  int tries = 0;
  cout << endl;
  do {
    programData.houghLinesPThreshold =
        (programData.houghLinesPThreshold + tries * 5) %
        MAX_HOUGH_LINES_P_THRESHOLD;
    HoughLinesP(result, rawLines, 1, CV_PI / 180,
                programData.houghLinesPThreshold, 30, 10);

    cout << "try: " << tries << ", rawLines.size() = " << rawLines.size() << endl;

    vector<Line> lines;

    for (auto &rawLine : rawLines) {
      lines.push_back(Line(rawLine));
    }

    mergedClockLines = selectLinesCloseToCircleCenter(
        lines, clockCircle, DEFAULT_LINES_SELECTION_RADIUS_FACTOR);

    cout << "mergedClockLines.size() = " << mergedClockLines.size() << ", after selectLinesCloseToCircleCenter" << endl;

    //mergedClockLines = clockPointerLinesMerge(mergedClockLines, DEFAULT_LINES_MERGE_ANGLE, circle);

    cout << "mergedClockLines.size() = " << mergedClockLines.size() << ", after lineOfSymmetryClockPointerLinesMerge" << endl;

    if (rawLines.size() >= 2) break;
  } while (++tries < 5);

  return mergedClockLines;
}

void isolateClock(Circle &clockCircle, Mat &image, Mat &clock) {
  cv::Mat mask = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);
  auto clockCircleRadiusInt = static_cast<int>(clockCircle.radius);
  circle(mask, clockCircle.center, clockCircleRadiusInt, Scalar(255, 255, 255),
         -1, LINE_8, 0);
  Mat temp;
  image.copyTo(temp, mask);

  // Setup a rectangle to define your region of interest
  cv::Rect2d myROI(clockCircle.center.x - clockCircle.radius,
                   clockCircle.center.y - clockCircle.radius,
                   2.0 * clockCircle.radius, 2.0 * clockCircle.radius);
  clockCircle.center.x = clockCircle.radius;
  clockCircle.center.y = clockCircle.radius;

  // Crop the full image to that image contained by the rectangle myROI
  // Note that this doesn't copy the data
  clock = temp(myROI);
}

vector<Line> selectLinesCloseToCircleCenter(vector<Line> &lines,
                                            const Circle &circle,
                                            double radiusFactor) {
  double clock_radius_limit = radiusFactor * circle.radius;
  vector<Line> clockPointerLines;

  for (auto &line : lines) {
    Point2d vec1 = line.a - circle.center;
    Point2d vec2 = line.b - circle.center;

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

vector<Line> clockPointerLinesMerge(vector<Line> &clockLines, int linesMergeAngle, Circle &clockCircle) {
  vector<Line> result;

  size_t clockLinesSize = clockLines.size();
  if (clockLinesSize == 0) {
    return result;
  }

  for (size_t x = 0; x < clockLinesSize - 1; x++) {
    Line l1 = clockLines[x];

    Point vec1 = calcLineVec(l1);

    bool l1Merged = false;
    for (size_t y = x + 1; y < clockLinesSize; y++) {
      Line l2 = clockLines[y];

      Point vec2 = calcLineVec(l2);

      double vec1Vec2Angle = angleBetweenTwoLines(vec1, vec2, false);
      if (vec1Vec2Angle < linesMergeAngle) {
        l1Merged = true;
        double maxNorm = max(norm(vec1), norm(vec2));
        double newLineAngle = vec1Vec2Angle/2;
        auto newPointB = Point2d(maxNorm * cos(newLineAngle), maxNorm * sin(newLineAngle));
        Line newLine(clockCircle.center, newPointB);
        if (result.size() > x) {
          result[x] = newLine;
        } else {
          result.push_back(newLine);
        }
      }
    }
    if (!l1Merged) {
      result.push_back(l1);
    }
  }
  return result;
}

TimeExtracted extractTime(const vector<Line> &mergedClockLines,
                          const Circle &circle) {
  // we have 1 or two lines
  if (mergedClockLines.size() <= 0) return TimeExtracted();

  // determine mid night clock pointer to help determine the corret hour and
  // minute
  Point2d limitPoint;
  limitPoint.x = circle.center.x;
  limitPoint.y = circle.center.y - circle.radius;
  Line midNightLine = Line(circle.center, limitPoint);
  Point2d vecReference = midNightLine.b - midNightLine.a;

  // get the first clock pointer
  Line pointer_1 = mergedClockLines[0];
  Point2d vecPointer_1 = pointer_1.b - pointer_1.a;
  double ang_1 = clockWiseAngleBetweenTwoVectors(vecReference, vecPointer_1);

  if (mergedClockLines.size() >= 2) {
    // get the secound clock pointer
    Line pointer_2 = mergedClockLines[1];
    Point2d vecPointer_2 = pointer_2.b - pointer_2.a;
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
int getHourFromAngleDeg(double angle) {
  return static_cast<int>((angle * 12.0) / 360.0);
}

int getMinuteFromAngleDeg(double angle) {
  return static_cast<int>((angle * 60.0) / 360.0);
}

ostream &operator<<(ostream &ostr, const TimeExtracted &time) {
  ostr << "Hours: " << time.hour << ":" << time.minute;
  return ostr;
}

double clockWiseAngleBetweenTwoVectors(const Point2d &vec1,
                                       const Point2d &vec2) {
  if (vec1 == vec2) {
    return 0;
  }
  double dot = vec1.dot(vec2);
  double det = vec1.x * vec2.y - vec1.y * vec2.x;  // determinant
  double ang = atan2(det, dot);  // atan2(y, x) or atan2(sin, cos)
  if (ang < 0) {
    ang = 2 * M_PI + ang;
  }
  return (ang * 180.0) / M_PI;
}

// TODO: verify if one of the norms is zero...
double angleBetweenTwoLines(const Point2d &vec1, const Point2d &vec2, bool toDegree) {
  if (vec1 == vec2) {
    return 0;
  }
  double ang = acos(vec1.dot(vec2) / (norm(vec1) * norm(vec2)));

  if (toDegree) {
    return (ang * 180.0) / M_PI;
  }

  return ang;
}

SegmentsType segmentsAnalysis(Line &line1, Line &line2,
                              Point2d &intersectingPoint, double doubleEqualityInterval) {
  Point2d q = line2.a;
  Point2d p = line1.a;
  Point2d qMp = q - p;
  Point2d r = calcLineVec(line1);
  Point2d s = calcLineVec(line2);
  double rXs = crossProduct2d(r, s);
  double qMpXr = crossProduct2d(qMp, r);

  if (doubleIsZero(rXs, doubleEqualityInterval) &&
      doubleIsZero(qMpXr, doubleEqualityInterval)) {
    double rN = norm(r);
    double rNSquared = rN * rN;
    double t0 = qMp.dot(r) / rNSquared;
    double t1 = t0 + s.dot(r) / rNSquared;
    if (s.dot(r) < 0) {
      swap(t0, t1);
    }
    if (intervalsIntersect(t0, t1, 0, 1)) {
      return SegmentsType::COLLINEAR_OVERLAPPING;
    } else {
      return SegmentsType::COLLINEAR_DISJOINT;
    }
  }

  if (doubleIsZero(rXs, doubleEqualityInterval) &&
      !doubleIsZero(qMpXr, doubleEqualityInterval)) {
    return SegmentsType::PARALLEL;
  }

  if (!doubleIsZero(rXs, doubleEqualityInterval)) {
    double t = crossProduct2d(qMp, s) / rXs;
    double u = qMpXr / rXs;
    if (t >= 0 && t <= 1 && u >= 0 && u <= 1) {
      calcPointDisplacement(p, r, t, intersectingPoint);
      return SegmentsType::SEGMENTS_INTERSECTING;
    }
  }

  return SegmentsType::SEGMENTS_NOT_INTERSECTING;
}

bool intervalsIntersect(double t0, double t1, double x0, double x1) {
  return max(t0, x0) <= min(t1, x1);
}

double crossProduct2d(Point2d &vec1, Point2d &vec2) {
  return vec1.x * vec2.y - vec1.y * vec2.x;
}

Point2d calcLineVec(Line &line) { return line.b - line.a; }

bool doubleIsZero(double value, double interval) {
  return doubleEquality(value, 0, interval);
}

bool doubleEquality(double value, double reference, double interval) {
  return value > reference - interval && value < reference + interval;
}

void calcPointDisplacement(Point2d &start, Point2d &displacementVector,
                           double displacementFactor, Point2d &displacedPoint) {
  displacedPoint.x = start.x + displacementVector.x * displacementFactor;
  displacedPoint.y = start.y + displacementVector.y * displacementFactor;
}

void swapPoints(Line &line) { swap(line.a, line.b); }

/*
 * Calculates the median color of a image from hist
 * TODO: Canny betweem 0.66*[median value] and 1.33*[median value]
 */
double medianHist(Mat grayImage) {
  // Initialize parameters
  int histSize = 256;  // bin size
  float range[] = {0, 255};
  const float *ranges[] = {range};

  // Calculate histogram
  MatND hist;
  calcHist(&grayImage, 1, nullptr, Mat(), hist, 1, &histSize, ranges, true,
           false);

  double m = grayImage.rows * grayImage.cols / 2;
  int bin = 0;
  double med = -1.0;

  for (int i = 0; i < histSize && med < 0.0; ++i) {
    bin += cvRound(hist.at<float>(i));
    if (bin > m && med < 0.0) med = i;
  }

  cout << "Median " << endl;

  return med;
}

string readCommandLine(int argc, char **argv, string const &defaultImagePath) {
  string imagePath = defaultImagePath;
  if (argc > 1) {
    imagePath = argv[1];
  }
  return imagePath;
}

void readImage(string &imagePath, ProgramData &programData) {
  programData.origImg = imread(imagePath, IMREAD_COLOR);
  bgr2gray(programData.origImg, programData.grayImage);

  if (programData.origImg.empty()) {
    throw std::invalid_argument("Invalid Input Image");
  }
}

void buildGui(TrackbarCallback callback, ProgramData &programData) {
  createWindow(WINDOW_NAME, programData.origImg.rows, programData.origImg.cols);

  createTrackbar(HOUGH_CIRCLES_CANNY_THRESHOLD_TRACKBAR_NAME, WINDOW_NAME,
                 &programData.houghCirclesCannyThreshold,
                 MAX_HOUGH_CIRCLES_CANNY_THRESHOLD, callback, &programData);
  createTrackbar(HOUGH_CIRCLES_ACCUMULATOR_THRESHOLD_TRACKBAR_NAME, WINDOW_NAME,
                 &programData.houghCirclesAccumulatorThreshold,
                 MAX_HOUGH_CIRCLES_ACCUMULATOR_THRESHOLD, callback,
                 &programData);
  createTrackbar(BILATERAL_SIGMA_COLOR_TRACKBAR_NAME, WINDOW_NAME,
                 &programData.bilateralSigmaColor, MAX_BILATERAL_SIGMA,
                 callback, &programData);
  createTrackbar(BILATERAL_SIGMA_SPACE_TRACKBAR_NAME, WINDOW_NAME,
                 &programData.bilateralSigmaSpace, MAX_BILATERAL_SIGMA,
                 callback, &programData);
  createTrackbar(CANNY_THRESHOLD1_TRACKBAR_NAME, WINDOW_NAME,
                 &programData.cannyThreshold1, MAX_CANNY_TRESHOLD, callback,
                 &programData);
  createTrackbar(CANNY_THRESHOLD2_TRACKBAR_NAME, WINDOW_NAME,
                 &programData.cannyThreshold2, MAX_CANNY_TRESHOLD, callback,
                 &programData);
  createTrackbar(DOUBLE_EQUALITY_INTERVAL_RADIUS_PERCENTAGE_TRACKBAR_NAME, WINDOW_NAME,
                 &programData.doubleEqualityIntervalRadiusPercentage, MAX_DOUBLE_EQUALITY_INTERVAL_RADIUS_PERCENTAGE, callback,
                 &programData);
}

void createWindow(string windowName, int rows, int cols) {
  if (rows > 600 || cols > 600) {
    namedWindow(windowName, WINDOW_NORMAL);
    resizeWindow(windowName, 600, 600);
  } else {
    namedWindow(windowName, WINDOW_AUTOSIZE);
  }
}

void imageShow(string windowName, Mat &image) {
  createWindow(windowName, image.rows, image.cols);
  imshow(windowName, image);
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

Scalar getDistinctColor(size_t index, size_t numberOfDistinctColors) {
  Mat bgr;
  Mat hsv(1, 1, CV_8UC3,
          Scalar(static_cast<double>(index * 179 / numberOfDistinctColors), 255,
                 255));
  cvtColor(hsv, bgr, CV_HSV2BGR);
  return Scalar(bgr.data[0], bgr.data[1], bgr.data[2]);
}
