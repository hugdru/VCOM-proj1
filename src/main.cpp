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
const string DEFAULT_IMAGE_PATH = "../data/watch.jpg";

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

constexpr double DEFAULT_LINES_MERGE_ANGLE = 0.0874f; //5º rad, temp to sec 0.045f

constexpr double DEFAULT_LINES_SELECTION_RADIUS_FACTOR = 0.25;

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

enum class TimeLinesType {
  HOUR_MINUTE = 0,
  HOUR_MINUTE_SECOND,
  UNKNOWN
};

struct TimeLines {
  TimeLinesType timesLinesType;
  Line hour, minute, second;

  TimeLines() {
    timesLinesType = TimeLinesType::UNKNOWN;
  }

  TimeLines(Line hour_, Line minute_) {
    timesLinesType = TimeLinesType::HOUR_MINUTE;
    hour = hour_;
    minute = minute_;
  }

  TimeLines(Line hour_, Line minute_, Line second_) {
    timesLinesType = TimeLinesType::HOUR_MINUTE_SECOND;
    hour = hour_;
    minute = minute_;
    second = second_;
  }
};

struct TimeExtracted {
	int hour;
	int minute;
	int secs;

	TimeExtracted() : hour(0), minute(0), secs(0) {}
	explicit TimeExtracted(int h, int m) : hour(h), minute(m), secs(0) {}
	explicit TimeExtracted(int h, int m, int s) : hour(h), minute(m), secs(s) {}
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

TimeLines getPointerLines(Mat &result, ProgramData &programData,
	const Circle &clockCircle);

vector<Line> iterativeLinesSearch(ProgramData &programData, const Circle &clockCircle, Mat &image, size_t numberOfLinesToBreak);

void isolateClock(Circle &clockCircle, Mat &image, Mat &clock);

vector<Line> selectLinesCloseToCircleCenter(vector<Line> &lines,
	const Circle &circle,
	double radiusFactor);
vector<Line> clockPointerLinesMerge(vector<Line> clockLines, double linesMergeAngle, const Circle &clockCircle);

TimeExtracted extractTime(TimeLines &timeLines,
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
double getHourFromAngleDeg(double angle);
double getMinuteSecFromAngleDeg(double angle);
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
Mat calculateRedMask(Mat input);

int main(int argc, char **argv) {
	string imagePath = readCommandLine(argc, argv, DEFAULT_IMAGE_PATH);
	ProgramData programData = ProgramData();
	readImage(imagePath, programData);
	buildGui(clockTimeDetector, programData);
	clockTimeDetector(0, &programData);
	waitKey(0);
	return 0;
}

void clockTimeDetector(int, void *rawProgramData) {
	auto *programData = static_cast<ProgramData *>(rawProgramData);

	vector<Circle> circles = getCircles(*programData);
	if (circles.empty()) {
		return;
	}

  for (auto &clockCircle : circles) {
    isolateClock(clockCircle, programData->origImg, programData->imgCropped);
    bgr2gray(programData->imgCropped, programData->grayImageCropped);

	Mat SecPointerMask;
    Mat display;
    TimeLines timeLines =
        getPointerLines(display, *programData, clockCircle);

    if (timeLines.timesLinesType != TimeLinesType::UNKNOWN) {
      gray2bgr(display, display);

	  TimeExtracted hoursExtracted = extractTime(timeLines, clockCircle);

      cout << endl << hoursExtracted << endl;

      line(display, timeLines.hour.a, timeLines.hour.b, Scalar(255, 0, 0), 3, LINE_AA);
      line(display, timeLines.minute.a, timeLines.minute.b, Scalar(0, 255, 0), 3, LINE_AA);

      if (timeLines.timesLinesType == TimeLinesType::HOUR_MINUTE_SECOND) {
        line(display, timeLines.second.a, timeLines.second.b, Scalar(0, 0, 255), 3, LINE_AA);
      }

      Point2d limitPoint;
      limitPoint.x = clockCircle.center.x;
      limitPoint.y = clockCircle.center.y - clockCircle.radius;
      Line midNightLine = Line(clockCircle.center, limitPoint);
      line(display, midNightLine.a, midNightLine.b, Scalar(0, 255, 255), 3,
		   LINE_AA);
		
		// circle center
		circle(display, clockCircle.center, 3, Scalar(0, 255, 0), -1, 8, 0);
		// circle outline
		circle(display, clockCircle.center, DEFAULT_LINES_SELECTION_RADIUS_FACTOR * clockCircle.radius, Scalar(0, 0, 255), 3, 8, 0);

      imshow(WINDOW_NAME, display);

      break;
    }
	}
}

vector<Circle> getCircles(ProgramData &programData) {

	Mat blurredImage;
	GaussianBlur(programData.grayImage, blurredImage, Size(9, 9), 2, 2);

  int tries = 0;
  cout << endl;
  vector<Circle> circles;
  do {
    programData.houghCirclesCannyThreshold =
        max((DEFAULT_HOUGH_CIRCLES_CANNY_THRESHOLD - tries * 5), MIN_HOUGH_CIRCLES_CANNY_THRESHOLD);

    std::vector<Vec3f> raw_circles;

    /*
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
    */
    HoughCircles(blurredImage, raw_circles, HOUGH_GRADIENT, 1,
                 blurredImage.rows / 8, programData.houghCirclesCannyThreshold,
                 programData.houghCirclesAccumulatorThreshold, 0, 0);

    circles.clear();
    for (auto &raw_circle : raw_circles) {
      circles.push_back(Circle(raw_circle));
    }
    cout << "circles " << circles.size() << endl;
    if (!circles.empty()) {
      break;
    }
  } while (++tries < 50);

  std::sort(circles.begin(),
            circles.end(),
            [](Circle const &a, Circle const &b) -> bool { return a.radius > b.radius; });
  return circles;
}

TimeLines getPointerLines(Mat &result, ProgramData &programData,
	const Circle &clockCircle) {
	imageShow("before bilateral", programData.grayImageCropped);

	bilateralFilter(programData.grayImageCropped, result,
		programData.bilateralSigmaColor,
		programData.bilateralSigmaSpace, BORDER_DEFAULT);

	imageShow("after bilateral", result);

	Canny(result, result, programData.cannyThreshold1,
		programData.cannyThreshold2, programData.cannyApertureSize);

	imageShow("after canny", result);

	// extract red pointer of seconds
	Mat redMask = calculateRedMask(programData.imgCropped);
	int dilation_size = 2;
	imageShow("redmask without dilation", redMask);
	Mat element = getStructuringElement(MORPH_CROSS,
		Size(2 * dilation_size + 1, 2 * dilation_size + 1),
		Point(dilation_size, dilation_size));
	/// Apply the dilation operation
	dilate(redMask, redMask, element);
	imageShow("redmask with dilation", redMask);

    auto redMergedClockLines = iterativeLinesSearch(programData, clockCircle, redMask, 1);
    bool hasRedLine = !redMergedClockLines.empty();
    Line redLine;
    if (hasRedLine) {
      redLine = redMergedClockLines[0];
    }

  Mat imageHourSecond = result;
  if (hasRedLine) {
    result.setTo(cv::Scalar(0, 0, 0), redMask);
    imageShow("result after red mask", result);
  }

  auto mergedClockLines = iterativeLinesSearch(programData, clockCircle, imageHourSecond, 2);
  if (hasRedLine) {
    cout << "Encontrei red line!" << endl;
    if (mergedClockLines.size() >= 2) {
      return TimeLines(mergedClockLines[1], mergedClockLines[0], redLine);
    } else if (mergedClockLines.size() == 1) {
      return TimeLines(mergedClockLines[0], mergedClockLines[0], redLine);
    }
  } else {
    if (mergedClockLines.size() == 2) {
      return TimeLines(mergedClockLines[1], mergedClockLines[0]);
    } else if (mergedClockLines.size() == 1) {
      return TimeLines(mergedClockLines[0], mergedClockLines[0]);
    } else if (mergedClockLines.size() > 2) {
      return TimeLines(mergedClockLines[2], mergedClockLines[1], mergedClockLines[0]);
    }
  }

  return TimeLines();
}

vector<Line> iterativeLinesSearch(ProgramData &programData, const Circle &clockCircle, Mat &image, size_t numberOfLinesToBreak) {
  vector<Vec4d> rawLines;
  vector<Line> mergedClockLines;
  int tries = 0;
  cout << endl;
  do {
    programData.houghLinesPThreshold =
        (DEFAULT_HOUGH_LINES_P_THRESHOLD + tries * 5) %
            MAX_HOUGH_LINES_P_THRESHOLD;
    HoughLinesP(image, rawLines, 1, CV_PI / 180,
                programData.houghLinesPThreshold, 30, 10);

    cout << "try: " << tries << ", rawLines.size() = " << rawLines.size() << endl;

    vector<Line> lines;

    for (auto &rawLine : rawLines) {
      lines.push_back(Line(rawLine));
    }

    mergedClockLines = selectLinesCloseToCircleCenter(
        lines, clockCircle, DEFAULT_LINES_SELECTION_RADIUS_FACTOR);

    cout << "mergedClockLines.size() = " << mergedClockLines.size() << ", after selectLinesCloseToCircleCenter" << endl;

    mergedClockLines = clockPointerLinesMerge(mergedClockLines, DEFAULT_LINES_MERGE_ANGLE, clockCircle);

    cout << "mergedClockLines.size() = " << mergedClockLines.size() << ", after clockPointerLinesMerge" << endl;

    if (mergedClockLines.size() >= numberOfLinesToBreak) break;
  } while (++tries < 50);

  std::sort(mergedClockLines.begin(), mergedClockLines.end(), [](Line const &l1, Line const &l2) -> bool
  { return norm(l1.b - l1.a) > norm(l2.b - l2.a); } );

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

vector<Line> clockPointerLinesMerge(vector<Line> clockLines, double linesMergeAngle,const Circle &clockCircle) {
  vector<Line> result;

  if (clockLines.empty() || clockLines.size() == 1) {
    return clockLines;
  }

  for (size_t x = 0; x < clockLines.size() - 1; x++) {
    Line l1 = clockLines[x];

    Point2d vec1 = calcLineVec(l1);
    double maxNorm = norm(l1.b - clockCircle.center);
    double vec1Angle = atan2(vec1.y, vec1.x);
    double sumCos = cos(vec1Angle);
    double sumSin = sin(vec1Angle);
    for (size_t y = x + 1; y < clockLines.size(); y++) {
      Line l2 = clockLines[y];

      Point2d vec2 = calcLineVec(l2);

      double vec1Vec2Angle = angleBetweenTwoLines(vec1, vec2, false);
      cout << x << " - " << vec1Vec2Angle << " - " << linesMergeAngle << endl;
      if (vec1Vec2Angle < linesMergeAngle) {
        double vec2Angle = atan2(vec2.y, vec2.x);
        sumCos += cos(vec2Angle);
        sumSin += sin(vec2Angle);
        cout << x << " - joined " << endl;
        maxNorm = max(maxNorm, norm(l2.b - clockCircle.center));

        clockLines.erase(clockLines.begin() + y);
        y--;
      }else{
        cout << x << " - passed " << endl;
      }
	}
	clockLines.erase(clockLines.begin() + x);
	x--;

    double midAngle = atan2(sumSin, sumCos);
    cout << "midAngle: " << midAngle << " sumCos " << sumCos << " sumSin " << sumSin << endl;
    Point2d directionVector = Point2d(cos(midAngle), sin(midAngle)) * maxNorm;
    Point2d newPointB =  directionVector + clockCircle.center;

    Line clockAvgPointer(clockCircle.center, newPointB);
    result.push_back(clockAvgPointer);
    if (clockLines.empty())
      break;
  }
  if(clockLines.size() == 1){
		Line l1 = clockLines[0];
		Point2d vec1 = calcLineVec(l1);
		double maxNorm = norm(l1.b - clockCircle.center);
		double vec1Angle = atan2(vec1.y, vec1.x);
		cout << "last Pointer: " << vec1Angle << endl;
		Point2d directionVector = Point2d(cos(vec1Angle), sin(vec1Angle)) * maxNorm;
		Point2d newPointB = directionVector + clockCircle.center;
	
		Line clockAvgPointer(clockCircle.center, newPointB);
		result.push_back(clockAvgPointer);
  }
  return result;
}

TimeExtracted extractTime(TimeLines &timeLines,
	const Circle &circle) {

	if (timeLines.timesLinesType == TimeLinesType::UNKNOWN) return TimeExtracted();

	// determine mid night clock pointer to help determine the correct hour and minute
	Point2d limitPoint;
	limitPoint.x = circle.center.x;
	limitPoint.y = circle.center.y - circle.radius;
	Line midNightLine = Line(circle.center, limitPoint);
	Point2d vecReference = midNightLine.b - midNightLine.a;

    Line hourLine = timeLines.hour;
    Line minuteLine = timeLines.minute;

    Point2d hourVec = calcLineVec(hourLine);
    Point2d minuteVec = calcLineVec(minuteLine);

	double hourAng = clockWiseAngleBetweenTwoVectors(vecReference, hourVec);
	double minuteAng = clockWiseAngleBetweenTwoVectors(vecReference, minuteVec);

  double approximateHour = getHourFromAngleDeg(hourAng);
  double approximateMinute = getMinuteSecFromAngleDeg(minuteAng);
  double approximateSecond = 0.0;

  if (timeLines.timesLinesType == TimeLinesType::HOUR_MINUTE_SECOND) {
    Line secondLine = timeLines.second;
    Point2d secondVec = calcLineVec(secondLine);
    double secondAng = clockWiseAngleBetweenTwoVectors(vecReference, secondVec);
    approximateSecond = getMinuteSecFromAngleDeg(secondAng);
  }

    double hourDecimalPart = approximateHour - static_cast<int>(approximateHour);
    double minutePercentage = approximateMinute / 60.0;

    auto hourInt = static_cast<int>(approximateHour);
    auto minuteInt = static_cast<int>(approximateMinute);
    auto secondInt = static_cast<int>(approximateSecond);

    if (hourDecimalPart <= 0.15 && minutePercentage > 0.75) {
      hourInt = static_cast<int>(approximateHour) - 1;
    } else if (hourDecimalPart > 0.85 && minutePercentage < 0.25) {
      hourInt = (static_cast<int>(approximateHour) + 1) % 12;
    }

    if (hourInt == 0) {
      hourInt = 12;
    }

  return TimeExtracted(hourInt, minuteInt, secondInt);
}

// TODO: define for hours and degree
double getHourFromAngleDeg(double angle) {
  return angle * 12.0 / 360.0;
}

double getMinuteSecFromAngleDeg(double angle) {
  return angle * 60.0 / 360.0;
}

ostream &operator<<(ostream &ostr, const TimeExtracted &time) {
	ostr << "Hours: " << time.hour << ":" << time.minute << ":" << time.secs;
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
		}
		else {
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
	float range[] = { 0, 255 };
	const float *ranges[] = { range };

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
}

void createWindow(string windowName, int rows, int cols) {
	if (rows > 600 || cols > 600) {
		namedWindow(windowName, WINDOW_NORMAL);
		resizeWindow(windowName, 600, 600);
	}
	else {
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

Mat calculateRedMask(Mat input) {

	Mat hsv_image;

	cvtColor(input, hsv_image, cv::COLOR_BGR2HSV);
	Mat lower_red_hue_range;
	Mat upper_red_hue_range;
	inRange(hsv_image, cv::Scalar(0, 100, 100), cv::Scalar(10, 255, 255), lower_red_hue_range);
	inRange(hsv_image, cv::Scalar(150, 100, 100), cv::Scalar(179, 255, 255), upper_red_hue_range);

	Mat redMask = lower_red_hue_range | upper_red_hue_range;

	return redMask;

}