#pragma once

#include <iostream>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

using namespace std;
using namespace cv;

Point get_intersection(Vec4f line_1, Vec4f line_2);
double distance_to_intersection(Vec4f line, Point point_3);
void vanishing(Mat image, int iterations, int inner_threshold, int outer_threshold);
