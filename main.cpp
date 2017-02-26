/**
* @author Kyle
* @file main.cpp
* @brief Main file for image processing
*/

#include <iostream>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include "factor.h"
#include "vanishing.h"
#include "fundamental.h"

using namespace std;
using namespace cv;

// ----------------------------------- Main code ----------------------------------- //
int main(int argc, char** argv) {
	Mat image_1 = imread("data/1.jpg", IMREAD_GRAYSCALE);
	Mat image_1a = imread("data/1a.jpg", IMREAD_GRAYSCALE);
	Mat image_1b = imread("data/1b.jpg", IMREAD_GRAYSCALE);
	Mat image_2 = imread("data/2.jpg", IMREAD_GRAYSCALE);
	Mat image_2a = imread("data/2a.jpg", IMREAD_GRAYSCALE);
	Mat image_2b = imread("data/2b.jpg", IMREAD_GRAYSCALE);

	Mat f_image_1, f_image_2;
	resize(image_1a, f_image_1, Size(400, 400));
	resize(image_1b, f_image_2, Size(400, 400));
	// resize(image_2a, f_image_1, Size(400, 400));
	// resize(image_2b, f_image_2, Size(400, 400));

	vanishing(image_1, 10000, 100, 10000);
	// vanishing(image_2, 10000, 100, 10000);

	fundamental(f_image_1, f_image_2, 300, 500, 1);

	cv::waitKey();
	return 0;
}