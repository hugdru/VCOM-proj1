#ifndef VCOM_PROJ1_GRUPO10_HUGO_INES_PEDRO_CLOCK_HPP
#define VCOM_PROJ1_GRUPO10_HUGO_INES_PEDRO_CLOCK_HPP

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

void initiate(cv::Mat &guiImage);
void createWindow(std::string windowName, int rows, int cols);
void imageShow(std::string windowName, cv::Mat &image);

#endif  // VCOM_PROJ1_GRUPO10_HUGO_INES_PEDRO_CLOCK_HPP
