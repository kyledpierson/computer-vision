#include <iostream>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include "opencv2/xfeatures2d.hpp"
#include "opencv2/calib3d.hpp"

using namespace std;
using namespace cv;

/*
* Add a row to the input matrix, at the specified index
*/
void add_row(int j, int m, Mat A, vector<DMatch> matches,
	vector<KeyPoint> keypoints_1, vector<KeyPoint> keypoints_2) {
	int a = matches[m].queryIdx;
	int b = matches[m].trainIdx;
	Point q1 = keypoints_1[a].pt;
	Point q2 = keypoints_2[b].pt;

	A.at<double>(j, 0) = q1.x * q2.x;
	A.at<double>(j, 1) = q1.y * q2.x;
	A.at<double>(j, 2) = q2.x;
	A.at<double>(j, 3) = q1.x * q2.y;
	A.at<double>(j, 4) = q1.y * q2.y;
	A.at<double>(j, 5) = q2.y;
	A.at<double>(j, 6) = q1.x;
	A.at<double>(j, 7) = q1.y;
	A.at<double>(j, 8) = 1;
}

/*
 * Compare two matches based on distance
 */
bool match_compare(DMatch a, DMatch b) {
	return (a.distance < b.distance);
}

/*
 * Compute fundamental matrix between two images
 */
void fundamental(Mat image_1, Mat image_2, int num_matches, int iterations, double threshold) {
	Ptr<Feature2D> f2d = xfeatures2d::SIFT::create();

	// Step 1: Detect the keypoints
	vector<KeyPoint> keypoints_1, keypoints_2;
	f2d->detect(image_1, keypoints_1);
	f2d->detect(image_2, keypoints_2);

	// Step 2: Calculate descriptors (feature vectors)
	Mat descriptors_1, descriptors_2;
	f2d->compute(image_1, keypoints_1, descriptors_1);
	f2d->compute(image_2, keypoints_2, descriptors_2);

	// Step 3: Matching descriptor vectors using BFMatcher
	BFMatcher matcher = BFMatcher(NORM_L2, true);
	vector<DMatch> matches, inliers;
	vector<DMatch>::iterator match_it;
	matcher.match(descriptors_1, descriptors_2, matches);
	sort(matches.begin(), matches.end(), match_compare);
	matches.resize(num_matches);

	// ----- Fundamental Matrix ----- //
	int max = 0;
	Mat F_mat = Mat(3, 3, CV_64F);

	// Loop a certain number of times
	int progress = iterations / 50;
	for (int i = 0; i < iterations; ++i) {
		if (i % progress == 1) {
			cout << "=";
		}

		int new_max = 0;
		vector<DMatch> new_inliers = vector<DMatch>();

		// Get the 8 correspondences
		Mat A = Mat(8, 9, CV_64F);
		for (int j = 0; j < 8; ++j) {
			int m = rand() % matches.size();
			add_row(j, m, A, matches, keypoints_1, keypoints_2);
		}

		// Compute the fundamental matrix
		SVD sol = SVD(A);
		Mat V = Mat(9, 8, CV_64F);
		transpose(sol.vt, V);
		Mat F = V.col(V.cols - 1);

		// Check against other matches
		int counted = 0;
		for (int j = 0; j < matches.size(); j += 8) {
			for (int k = 0; k < 8; ++k) {
				if (j + k < matches.size()) {
					add_row(k, j + k, A, matches, keypoints_1, keypoints_2);
				}
				else {
					counted++;
					add_row(k, k, A, matches, keypoints_1, keypoints_2);
				}
			}

			Mat b = A * F;
			for (int k = 0; k < b.rows; ++k) {
				if (abs(b.at<double>(k, 0)) < threshold) {
					new_max++;
					if ((j + k) < matches.size()) {
						new_inliers.push_back(matches[j + k]);
					}
				}
			}
		}
		new_max = new_max - counted;

		// Update the matrix if more inliers
		if (new_max > max) {
			max = new_max;
			inliers = new_inliers;
			F_mat = F.clone().reshape(0, 3);
			F_mat.at<double>(2, 2) = abs(F_mat.at<double>(2, 2));
		}
	}

	// Validate fundamental matrix
	vector<Point2f> points_1 = vector<Point2f>();
	vector<Point2f> points_2 = vector<Point2f>();
	for (int i = 0; i < matches.size(); ++i) {
		points_1.push_back(keypoints_1[matches[i].queryIdx].pt);
		points_2.push_back(keypoints_2[matches[i].trainIdx].pt);
	}
	Mat fundamental_matrix = findFundamentalMat(points_1, points_2);

	// Show results
	Mat result;
	drawMatches(image_1, keypoints_1, image_2, keypoints_2, inliers, result);
	imshow("Matching Features", result);

	cout << endl << "Computed" << endl << F_mat << endl;
	cout << endl << "Actual" << endl << fundamental_matrix << endl;
	cout << endl << "Difference between matrices" << endl << norm(fundamental_matrix - F_mat) << endl;
}