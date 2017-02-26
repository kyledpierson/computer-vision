#include <iostream>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

using namespace std;
using namespace cv;

/*
* Get the intersection between two lines
*/
Point get_intersection(Vec4f line_1, Vec4f line_2) {
	double x1 = line_1[0]; double y1 = line_1[1]; double x2 = line_1[2]; double y2 = line_1[3];
	double x3 = line_2[0]; double y3 = line_2[1]; double x4 = line_2[2]; double y4 = line_2[3];

	double num_1 = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4));
	double num_2 = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4));
	double denominator = ((x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4));
	if (denominator == 0) {
		denominator = 0.1;
	}

	Point vec = Point(num_1 / denominator, num_2 / denominator);
	return vec;
}

/*
* Get the distance between a line and a point
*/
double distance_to_intersection(Vec4f line, Point point_3) {
	Point point_1 = Point(line[0], line[1]);
	Point point_2 = Point(line[2], line[3]);

	double denominator = sqrt((point_2.y - point_1.y) ^ 2 + (point_2.x - point_1.x) ^ 2);
	if (denominator == 0) {
		denominator = 0.1;
	}
	double result = abs((point_2.y - point_1.y)*point_3.x - (point_2.x - point_1.x)*point_3.y
		+ point_2.x*point_1.y - point_2.y*point_1.x) / denominator;

	return (unsigned)(int)result;
}

/*
 * Compute and show the vanishing points of the image
 */
void vanishing(Mat image, int iterations, int inner_threshold, int outer_threshold) {

#if 0
	Canny(image, image, 50, 200, 3);
#endif

#if 1
	Ptr<LineSegmentDetector> ls = createLineSegmentDetector(LSD_REFINE_STD);
#else
	Ptr<LineSegmentDetector> ls = createLineSegmentDetector(LSD_REFINE_NONE);
#endif

	// Detect the lines
	vector<Vec4f> lines_std;
	ls->detect(image, lines_std);

	int max_inliers_1 = 0;
	int max_inliers_2 = 0;
	int max_inliers_3 = 0;

	Point vanishing_point_1, vanishing_point_2, vanishing_point_3;
	vector<Vec4f> lines_inliers_1, lines_inliers_2, lines_inliers_3;

	// Loop a certain number of times
	int progress = iterations / 50;
	for (int i = 0; i < iterations; ++i) {
		if (i % progress == 1) {
			cout << "=";
		}

		int idx_1 = rand() % lines_std.size();
		int idx_2 = rand() % lines_std.size();
		Vec4f line_1 = lines_std[idx_1];
		Vec4f line_2 = lines_std[idx_2];
		Point intersection = get_intersection(line_1, line_2);

		vector<Vec4f>::iterator line_it;
		vector<Vec4f> lines_vanish = vector<Vec4f>{ line_1 , line_2 };
		for (line_it = lines_std.begin(); line_it != lines_std.end(); ++line_it) {
			if (distance_to_intersection(*line_it, intersection) < inner_threshold) {
				lines_vanish.push_back(*line_it);
			}
		}

		// Set inliers
		if (lines_vanish.size() > max_inliers_1) {
			max_inliers_1 = lines_vanish.size();
			lines_inliers_1 = lines_vanish;
			vanishing_point_1 = intersection;
		}
		else if (lines_vanish.size() > max_inliers_2
			&& norm(intersection - vanishing_point_1) > outer_threshold) {
			max_inliers_2 = lines_vanish.size();
			lines_inliers_2 = lines_vanish;
			vanishing_point_2 = intersection;
		}
		else if (lines_vanish.size() > max_inliers_3
			&& norm(intersection - vanishing_point_1) > outer_threshold
			&& norm(intersection - vanishing_point_2) > outer_threshold) {
			max_inliers_3 = lines_vanish.size();
			lines_inliers_3 = lines_vanish;
			vanishing_point_3 = intersection;
		}
	}

	// Show found lines
	Mat drawn_lines_1(image);
	Mat drawn_lines_2(image);
	Mat drawn_lines_3(image);

	ls->drawSegments(drawn_lines_1, lines_inliers_1);
	ls->drawSegments(drawn_lines_2, lines_inliers_2);
	ls->drawSegments(drawn_lines_3, lines_inliers_3);

	imshow("First Vanishing Inliers", drawn_lines_1);
	imshow("Second Vanishing Inliers", drawn_lines_2);
	imshow("Third Vanishing Inliers", drawn_lines_3);
	cout << endl << vanishing_point_1 << endl << vanishing_point_2 << endl << vanishing_point_3 << endl;
}