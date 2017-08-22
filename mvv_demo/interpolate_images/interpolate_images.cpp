#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <vector>
#include <ctime>

#include "interpolate_images.h"

using namespace cv;
using namespace std;

Mat affine_transform(const Vec6f & triangle_1, const Vec6f & triangle_2) {
	cv::Point2f source_triangle[3];
	source_triangle[0] = cv::Point2f(triangle_1[0], triangle_1[1]);
	source_triangle[1] = cv::Point2f(triangle_1[2], triangle_1[3]);
	source_triangle[2] = cv::Point2f(triangle_1[4], triangle_1[5]);
	cv::Point2f target_triangle[3];
	target_triangle[0] = cv::Point2f(triangle_2[0], triangle_2[1]);
	target_triangle[1] = cv::Point2f(triangle_2[2], triangle_2[3]);
	target_triangle[2] = cv::Point2f(triangle_2[4], triangle_2[5]);
	cv::Mat trans = cv::getAffineTransform(source_triangle, target_triangle);
	return trans;

	/*
	Parametrization should go:

	A = [(1-t) + a_00 * t, a_01 * t,
	      a_10 * t, (1-t) + a_11 * t]
	B = [t * b_0, t * b_1]
	*/
}

vector<Mat> get_affine_transforms_forward(const vector<Vec6f> &triangle_list_1, const vector<Vec6f> &triangle_list_2) {
	// Start off by calculating affine transformation of two triangles.
	int num_zeros = 0;

	int num_triangles = triangle_list_1.size();

	vector<Mat> transforms = vector<Mat>();
	for (int i = 0; i < num_triangles; i++) {
		Vec6f triangle_1 = triangle_list_1[i];
		Vec6f triangle_2 = triangle_list_2[i];
		cv::Point2f triangle_1_points[3];
		triangle_1_points[0] = cv::Point2f(triangle_1[0], triangle_1[1]);
		triangle_1_points[1] = cv::Point2f(triangle_1[2], triangle_1[3]);
		triangle_1_points[2] = cv::Point2f(triangle_1[4], triangle_1[5]);
		cv::Point2f triangle_2_points[3];
		triangle_2_points[0] = cv::Point2f(triangle_2[0], triangle_2[1]);
		triangle_2_points[1] = cv::Point2f(triangle_2[2], triangle_2[3]);
		triangle_2_points[2] = cv::Point2f(triangle_2[4], triangle_2[5]);
		cv::Mat trans = cv::getAffineTransform(triangle_1_points, triangle_2_points);
		if (trans.at<double>(0, 0) == 0 && trans.at<double>(0, 1) == 0 && trans.at<double>(0, 2) == 0 &&
			trans.at<double>(1, 0) == 0 && trans.at<double>(1, 1) == 0 && trans.at<double>(1, 2) == 0) {
			num_zeros++;
		}
		transforms.push_back(trans);
	}
	cout << "Number of forward null transforms: " << num_zeros << endl;
	return transforms;
}

void reverse_transform(const Mat &forward, Mat &reverse) {
	// Simply put, this reverses the transform using the formula:
	// X' = AX+B, so
	// X = A^(-1)X - A^(-1)B
	double a = forward.at<double>(0, 0);
	double b = forward.at<double>(0, 1);
	double c = forward.at<double>(1, 0);
	double d = forward.at<double>(1, 1);
	double b00 = forward.at<double>(0, 2);
	double b11 = forward.at<double>(1, 2);

	double det = a*d - b*c;
	if (det != 0) {
		double inv_det = (double)1 / det;
		// check for invertibility
		double inv_a = inv_det * d;
		double inv_b = -1 * inv_det * b;
		double inv_c = -1 * inv_det * c;
		double inv_d = inv_det * a;
		double inv_b00 = inv_a * b00 + inv_b * b11;
		double inv_b11 = inv_c * b00 + inv_d * b11;
		reverse.at<double>(0, 0) = inv_a;
		reverse.at<double>(0, 1) = inv_b;
		reverse.at<double>(0, 2) = inv_b00;
		reverse.at<double>(1, 0) = inv_c;
		reverse.at<double>(1, 1) = inv_d;
		reverse.at<double>(1, 2) = inv_b11;
	}
	else {
		cout << "Invalid determinant" << endl;
	}
}

vector<Mat> get_affine_transforms_reverse(const vector<Vec6f> &triangle_list_1, const vector<Vec6f> &triangle_list_2, const vector<Mat> &forward_transforms) {
	// Start off by calculating affine transformation of two triangles.
	int num_zeros = 0;

	int num_triangles = triangle_list_1.size();

	vector<Mat> transforms = vector<Mat>();
	for (int i = 0; i < num_triangles; i++) {
		Vec6f triangle_1 = triangle_list_1[i];
		Vec6f triangle_2 = triangle_list_2[i];
		cv::Point2f triangle_1_points[3];
		triangle_1_points[0] = cv::Point2f(triangle_1[0], triangle_1[1]);
		triangle_1_points[1] = cv::Point2f(triangle_1[2], triangle_1[3]);
		triangle_1_points[2] = cv::Point2f(triangle_1[4], triangle_1[5]);
		cv::Point2f triangle_2_points[3];
		triangle_2_points[0] = cv::Point2f(triangle_2[0], triangle_2[1]);
		triangle_2_points[1] = cv::Point2f(triangle_2[2], triangle_2[3]);
		triangle_2_points[2] = cv::Point2f(triangle_2[4], triangle_2[5]);
		cv::Mat trans = cv::getAffineTransform(triangle_1_points, triangle_2_points);
		if (trans.at<double>(0, 0) == 0 && trans.at<double>(0, 1) == 0 && trans.at<double>(0, 2) == 0 &&
			trans.at<double>(1, 0) == 0 && trans.at<double>(1, 1) == 0 && trans.at<double>(1, 2) == 0) {
			reverse_transform(forward_transforms[i], trans);
			num_zeros++;
		}
		transforms.push_back(trans);
	}
	cout << "Number of reverse null transforms: " << num_zeros << endl;
	return transforms;
}

vector<Vec6f> get_interpolated_triangles(const vector<Vec6f> &triangle_list_1, const vector<vector<vector<double>>> & affine, const int tInt) {
	int num_triangles = triangle_list_1.size();
	float t = (float)tInt / 100;
	vector<Vec6f> inter_triangles = vector<Vec6f>();

	// It is by will alone I set my mind in motion.
	// Good luck with these indices...
	for (int i = 0; i < num_triangles; i++) {
		vector<vector<double>> affine_params = affine[i];
		float pt1x = (1 - t + affine_params[0][0] * t) * triangle_list_1[i][0] + (affine_params[0][1] * t) * triangle_list_1[i][1] + (affine_params[0][2] * t);
		float pt1y = (affine_params[1][0] * t) * triangle_list_1[i][0] + (1 - t + affine_params[1][1] * t) * triangle_list_1[i][1] + (affine_params[1][2] * t);
		float pt2x = (1 - t + affine_params[0][0] * t) * triangle_list_1[i][2] + (affine_params[0][1] * t) * triangle_list_1[i][3] + (affine_params[0][2] * t);
		float pt2y = (affine_params[1][0] * t) * triangle_list_1[i][2] + (1 - t + affine_params[1][1] * t) * triangle_list_1[i][3] + (affine_params[1][2] * t);
		float pt3x = (1 - t + affine_params[0][0] * t) * triangle_list_1[i][4] + (affine_params[0][1] * t) * triangle_list_1[i][5] + (affine_params[0][2] * t);
		float pt3y = (affine_params[1][0] * t) * triangle_list_1[i][4] + (1 - t + affine_params[1][1] * t) * triangle_list_1[i][5] + (affine_params[1][2] * t);
		inter_triangles.push_back(Vec6f(pt1x, pt1y, pt2x, pt2y, pt3x, pt3y));
	}
	return inter_triangles;
}

void display_interpolated_triangles(const vector<Vec6f> & triangles, const Rect & image_bounds) {
	// the graphical_triangulation function is far too slow

	Scalar active_facet_color(0, 0, 255), delaunay_color(255, 255, 255);
	Mat img(image_bounds.size(), CV_8UC3);

	img = Scalar::all(0);
	string win = "Delaunay Demo";

	int num_triangles = triangles.size();
	for (size_t i = 0; i < num_triangles; i++)
	{
		Vec6f t = triangles[i];
		Point pt0 = Point(cvRound(t[0]), cvRound(t[1]));
		Point pt1 = Point(cvRound(t[2]), cvRound(t[3]));
		Point pt2 = Point(cvRound(t[4]), cvRound(t[5]));
		line(img, pt0, pt1, active_facet_color, 1, LINE_AA, 0);
		line(img, pt1, pt2, active_facet_color, 1, LINE_AA, 0);
		line(img, pt2, pt0, active_facet_color, 1, LINE_AA, 0);
	}

	imshow(win, img);
	waitKey(1);
}

struct interpolationMorph {
	Rect imageSize;
	int tInt;
	vector<Vec6f> sourceT;
	vector<Vec6f> targetT;
	vector<vector<vector<double>>> affineParams;
};

static void onInterpolate(int tInt, void *userdata) //void* mean that it is a pointer of unknown type
{
	(*((interpolationMorph*)userdata)).tInt = tInt;

	vector<Vec6f> sourceT = (*((interpolationMorph*)userdata)).sourceT;
	vector<Vec6f> targetT = (*((interpolationMorph*)userdata)).targetT;
	vector<vector<vector<double>>> affineParams = (*((interpolationMorph*)userdata)).affineParams;
	Rect imgSize = (*((interpolationMorph*)userdata)).imageSize;
	vector<Vec6f> interT = get_interpolated_triangles(sourceT, affineParams, tInt);
	display_interpolated_triangles(interT, imgSize);
	//Subdiv2D subdiv = raw_triangulation(interPoints, imgSize);
	//display_triangulation(subdiv, imgSize);
}

void interpolation_trackbar(const vector<Vec6f> & triangle_list_1, const vector<Vec6f> & triangle_list_2, const Rect & img1_size, const Rect & img2_size, const vector<vector<vector<double>>> & affine)
{
	// max of height and weidth
	int maxWidth = max(img1_size.width, img2_size.width);
	int maxHeight = max(img1_size.height, img2_size.height);
	Rect imgSize = Rect(0,0,maxWidth, maxHeight);

	interpolationMorph holder;
	holder.sourceT = triangle_list_1;
	holder.targetT = triangle_list_2;
	holder.imageSize = imgSize;
	holder.affineParams = affine;
	holder.tInt = 0;

	int tInt = 0;

	namedWindow("Adjust Window");
	cvCreateTrackbar2("Morph", "Adjust Window", &tInt, 100, onInterpolate, (void*)(&holder));
	waitKey(0);
}