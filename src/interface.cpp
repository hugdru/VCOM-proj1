#define _USE_MATH_DEFINES
#include <algorithm>
#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>

#include "clock.hpp"

using namespace cv;
using namespace std;


const string LOGO_PATH = "../data/logo.jpg";

const int DEFAULT_CAMERA = 0y;

const int NO_OPER = -1;
const int UPLOAD_PHOTO_OPER = 0;
const int TAKE_VIDEO_PHOTO_OPER = 1;
const int CALCULATE_CLOCK_OPER = 2;

Mat3b canvas;
string buttonUploadText("Upload photo");
string buttonVideoText("Take photo");
string buttonClockText("Calculate time");
string winName = "My cool GUI v0.1";
int oper = NO_OPER;
int state = 0;

Rect buttonUpload;
Rect buttonVideo;
Rect buttonClock;

Mat input;

void askForFile() {
	ifstream infile;

	cout << "Please enter the input file name> " << flush;
	while (true)
	{
		string imagePath;
		getline(cin, imagePath);
		input = imread(imagePath.c_str(), IMREAD_COLOR);

		if (!input.empty()) {
			imshow("Image", input);
			state = 0;
			break;
		}
		cout << "Invalid file. Please enter a valid input file name> " << flush;
	}
}

int TakeOneFrame()
{
	VideoCapture cap(DEFAULT_CAMERA);

	// Check if camera opened successfully
	if (!cap.isOpened()) {
		cout << "Error opening video stream or file" << endl;
		return -1;
	}

	Mat frame;

	while (true)
	{
		cap >> frame;
		frame.copyTo(input);
		imshow("Video", frame);
		if (char(waitKey(1)) == 'p') {
			imshow("Image", input);
			bool retake = false;

			while (!retake) {
				if (char(waitKey(1)) == 'y') { //Accept photo
					destroyWindow("Video Feed");
					state = 0; //accept new input
					return 0;
				}

				else if (char(waitKey(1)) == 'n') { //Retake photo
					retake = true;
					destroyWindow("Image");
				}

			}
		}

	}
}

int calculateClock() {
	state = 0; //accept new input
    initiate(input);
	return 0;
}

void callBackFunc(int event, int x, int y, int flags, void* userdata)
{
	if (event == EVENT_LBUTTONDOWN)
	{
		if (buttonUpload.contains(Point(x, y)))
		{
			cout << "Clicked Upload!" << endl;
			auto buttonUploadCanvas = canvas(buttonUpload);
			rectangle(buttonUploadCanvas, buttonUpload, Scalar(0, 0, 255), 2);
			oper = UPLOAD_PHOTO_OPER;
			state = 1;
		}

		if (buttonVideo.contains(Point(x, y)))
		{
			cout << "Clicked Video!" << endl;
			auto buttonVideoCanvas = canvas(buttonVideo);
			rectangle(buttonVideoCanvas, buttonVideo, Scalar(0, 0, 255), 2);
			oper = TAKE_VIDEO_PHOTO_OPER;
			state = 1;

		}

		if (buttonClock.contains(Point(x, y)))
		{
			cout << "Clicked Clock!" << endl;
			auto buttonClockCanvas = canvas(buttonClock);
			rectangle(buttonClockCanvas, buttonClock, Scalar(0, 0, 255), 2);
			oper = CALCULATE_CLOCK_OPER;
			state = 1;

		}
	}
	if (event == EVENT_LBUTTONUP && state == 1)
	{
		rectangle(canvas, buttonUpload, Scalar(200, 200, 200), 2);
		rectangle(canvas, buttonVideo, Scalar(200, 200, 200), 2);
		rectangle(canvas, buttonClock, Scalar(200, 200, 200), 2);

		state = 2;
	}

	imshow(winName, canvas);
	waitKey(1);

	if (oper == UPLOAD_PHOTO_OPER && state == 2) {
		askForFile();
		oper = NO_OPER;
	}
	else if (oper == TAKE_VIDEO_PHOTO_OPER && state == 2) {
		TakeOneFrame();
		oper = NO_OPER;
	}
	else if (oper == CALCULATE_CLOCK_OPER && state == 2) {
		calculateClock();
		oper = NO_OPER;
	}
}

int main()
{
	Mat img = imread(LOGO_PATH, IMREAD_COLOR);

	// Your button
	buttonUpload = Rect(0, 0, img.cols, 50);
	buttonVideo = Rect(0, 50 + 10, img.cols, 50);
	buttonClock = Rect(0, 100 + 20, img.cols, 50);

	// The canvas
	canvas = Mat3b(img.rows + 3 * buttonUpload.height + 2 * 10, img.cols, Vec3b(255, 255, 255));

	// Draw the buttons
	//--------------------------------------------------
	canvas(buttonUpload) = Vec3b(200, 200, 200);
	putText(canvas(buttonUpload), buttonUploadText, Point(buttonUpload.width*0.35, buttonUpload.height*0.7), FONT_HERSHEY_PLAIN, 1, Scalar(0, 0, 0));

	canvas(buttonVideo) = Vec3b(200, 200, 200);
	putText(canvas(buttonVideo), buttonVideoText, Point(buttonVideo.width*0.35, buttonVideo.height*0.7), FONT_HERSHEY_PLAIN, 1, Scalar(0, 0, 0));

	canvas(buttonClock) = Vec3b(200, 200, 200);
	putText(canvas(buttonClock), buttonClockText, Point(buttonClock.width*0.35, buttonClock.height*0.7), FONT_HERSHEY_PLAIN, 1, Scalar(0, 0, 0));

	//--------------------------------------------------

	// Draw the image
	img.copyTo(canvas(Rect(0, 3 * buttonUpload.height + 2 * 10, img.cols, img.rows)));

	// Setup callback function
	namedWindow(winName);
	setMouseCallback(winName, callBackFunc);

	imshow(winName, canvas);

	while (true) {
		char key = (char)cv::waitKey(30);   // explicit cast
		if (key == 27) break;                // break if `esc' key was pressed. 
	}

	return 0;
}