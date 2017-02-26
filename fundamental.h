#pragma once

#include <iostream>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include "opencv2/xfeatures2d.hpp"
#include "opencv2/calib3d.hpp"

using namespace std;
using namespace cv;

void add_row(int j, int m, Mat A, vector<DMatch> matches, vector<KeyPoint> keypoints_1, vector<KeyPoint> keypoints_2);
bool match_compare(DMatch a, DMatch b);
void fundamental(Mat image_1, Mat image_2, int num_matches, int iterations, double threshold);
