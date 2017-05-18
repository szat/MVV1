// core.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <ctime>
#include <stdlib.h>
#include <iostream>
#include <chrono>
#include <fstream>

#include "binary_io.h"

#include "build_geometry.h"
#include "generate_test_points.h"

#include "match_points.h"
#include "good_features.h"
#include "affine_akaze.h"
#include "knn_test.h"

#include "interpolate_images.h"
#include "polygon_raster.h"

#define VERSION "1.0.0"
#define APPLICATION_NAME "MVV"
#define COMPANY_NAME "NDim Inc."
#define COPYRIGHT_YEAR 2017
#define CLOCKS_PER_MS (CLOCKS_PER_SEC / 1000)
#define NUM_THREADS 16

typedef std::chrono::high_resolution_clock Clock;

using namespace std;
using namespace cv;

int corner_points_test() {
	cout << "Begin integration test of match_features and build_geometry" << endl;

	//this is the image used in trackbarCorners
	string image_path = "..\\data_store\\david_1.jpg";
	Mat src1 = imread(image_path, IMREAD_GRAYSCALE); 
	vector<Point2f> corners;
	trackbarCorners(image_path, corners);


	Rect test_rect = Rect(0, 0, src1.size().width, src1.size().height);
	graphical_triangulation(corners, test_rect);

	return 0;
}

Rect get_image_size(string imagePath) {
	string address = "..\\data_store\\" + imagePath;
	string input = "";
	ifstream infile1;
	infile1.open(address.c_str());

	Mat img1 = imread(address, IMREAD_GRAYSCALE);

	float height = img1.size().height;
	float width = img1.size().width;

	Rect image_size = Rect(0, 0, width, height);
	return image_size;
}

int test_matching() {

	string imageA = "david_1.jpg";
	string imageB = "david_2.jpg";

	Rect image_size_A = get_image_size(imageA);
	Rect image_size_B = get_image_size(imageB);

	vector<vector<KeyPoint>> point_matches = test_match_points(imageA, imageB);

	vector<KeyPoint> image_matchesA = point_matches.at(0);
	vector<KeyPoint> image_matchesB = point_matches.at(1);

	vector<Point2f> image_pointsA = vector<Point2f>();

	int vecSize = image_matchesA.size();

	for (int i = 0; i < vecSize; i++) {
		image_pointsA.push_back(image_matchesA.at(i).pt);
	}

	//graphical_triangulation(imagePointsA, imageSizeA);
	//Subdiv2D subdiv = raw_triangulation(imagePointsA, imageSizeA);
	//display_triangulation(subdiv, imageSizeA);

	return 0;
}

void image_diagnostics(Mat img) {
	Size size = img.size();
	cout << "Image width: " << size.width << endl;
	cout << "Image height: " << size.height << endl;
}

struct GeometricSlice {
	Rect img;
	vector<Vec6f> triangles;
};

struct MatchedGeometry {
	GeometricSlice source_geometry;
	GeometricSlice target_geometry;
};

vector<Vec6f> split_trapezoids(vector<pair<Vec4f, Vec4f>> trapezoids) {
	vector<Vec6f> triangles = vector<Vec6f>();
	int size = trapezoids.size();
	for (int i = 0; i < size; i++) {
		Point2f pointA = Point2f(trapezoids[i].first[0], trapezoids[i].second[0]);
		Point2f pointB = Point2f(trapezoids[i].first[1], trapezoids[i].second[1]);
		Point2f pointC = Point2f(trapezoids[i].first[2], trapezoids[i].second[2]);
		Point2f pointD = Point2f(trapezoids[i].first[3], trapezoids[i].second[3]);
		// ABC, ACD triangles
		Vec6f triangleA = Vec6f(pointA.x, pointA.y, pointB.x, pointB.y, pointC.x, pointC.y);
		Vec6f triangleB = Vec6f(pointA.x, pointA.y, pointC.x, pointC.y, pointD.x, pointD.y);
		triangles.push_back(triangleA);
		triangles.push_back(triangleB);
	}
	return triangles;
}

MatchedGeometry create_matched_geometry(vector<Point2f> imgA_points, vector<Point2f> imgB_points, Size size) {
	// triangulate source interior
	vector<Vec6f> trianglesA = construct_triangles(imgA_points, size);

	// triangulate target interior
	vector<Vec6f> trianglesB = triangulate_target(imgA_points, imgB_points, trianglesA);

	Rect img_bounds = Rect(0, 0, size.width, size.height);

	// This could potentially be replaced by two constructors.
	MatchedGeometry matched_result = MatchedGeometry();
	GeometricSlice source = GeometricSlice();
	GeometricSlice target = GeometricSlice();
	source.triangles = trianglesA;
	target.triangles = trianglesB;
	source.img = img_bounds;
	target.img = img_bounds;
	matched_result.source_geometry = source;
	matched_result.target_geometry = target;
	return matched_result;
}

MatchedGeometry read_matched_points_from_file(string source_path, string target_path) {
	// Please note that:
	// A: source image
	// B: target image
	// I use this shorthand so that the variable names are shorter.

	cout << "Initializing matched geometry routine" << endl;

	string imgA_path = source_path;
	string imgB_path = target_path;
	string root_path = "../data_store";
	Mat imgA = cv::imread(root_path + "/" + imgA_path, IMREAD_GRAYSCALE);
	Mat imgB = cv::imread(root_path + "/" + imgB_path, IMREAD_GRAYSCALE);

	Size desired_size = imgB.size();
	resize(imgA, imgA, desired_size);

	vector<vector<KeyPoint>> point_matches = match_points_mat(imgA, imgB);

	vector<KeyPoint> imgA_keypoints = point_matches[0];
	vector<KeyPoint> imgB_keypoints = point_matches[1];
	vector<Point2f> imgA_points = convert_key_points(imgA_keypoints);
	vector<Point2f> imgB_points = convert_key_points(imgB_keypoints);

	MatchedGeometry geometry = create_matched_geometry(imgA_points, imgB_points, desired_size);
	return geometry;
}

void render_matched_geometry(GeometricSlice slice, string window_name) {
	// Render both images.

	// Trapezoids in red (first)
	Mat img(slice.img.size(), CV_8UC3);
	Scalar trapezoid_color(0, 0, 255), triangle_color(255, 255, 255), hull_color(255, 0, 0);
	string win = window_name;

	vector<Point> pt(3);
	int num_triangles = slice.triangles.size();
	for (int i = 0; i < num_triangles; i++) {
		Vec6f t = slice.triangles[i];
		pt[0] = Point(cvRound(t[0]), cvRound(t[1]));
		pt[1] = Point(cvRound(t[2]), cvRound(t[3]));
		pt[2] = Point(cvRound(t[4]), cvRound(t[5]));
		line(img, pt[0], pt[1], triangle_color, 1, LINE_AA, 0);
		line(img, pt[1], pt[2], triangle_color, 1, LINE_AA, 0);
		line(img, pt[2], pt[0], triangle_color, 1, LINE_AA, 0);
	}

	imshow(win, img);
	waitKey(1);
	// Triangles in white (second)
	// Border (convex hull in blue) (last)
}

vector<Point2f> parametrized_interpolation(float t, vector<Point2f> points, double a00, double a01, double a10, double a11, double b00, double b01) {
	vector<Point2f> param_points = vector<Point2f>();
	for (int i = 0; i < 3; i++) {
		float xP = (1 - t + a00 * t) * points[i].x + (a01 * t) * points[i].y + (b00 * t);
		float yP = (a10 * t) * points[i].x + (1 - t + a11 * t) * points[i].y + (b01 * t);
		param_points.push_back(Point2f(xP, yP));
	}
	return param_points;
}

void set_mask_to_triangle(Mat &mask, Vec6f t) {
	Point pts[3] = {
		Point(t[0],t[1]),
		Point(t[2],t[3]),
		Point(t[4],t[5]),
	};
	fillConvexPoly(mask, pts, 3, Scalar(1));
}

void save_frame_at_tau(Mat &imgA, Mat &imgB, Rect img_rect,
	vector<Mat> & affine_forward, vector<Mat> & affine_reverse, 
	vector<Vec6f> & trianglesA, vector<Vec6f> & trianglesB, float tau) {

	int x_dim = img_rect.width;
	int y_dim = img_rect.height;
	
	Mat canvas = Mat::zeros(y_dim, x_dim, CV_8UC1);

	// get affine
	Mat current_maskA = cv::Mat::zeros(y_dim, x_dim, CV_8UC1);
	Mat current_maskB = cv::Mat::zeros(y_dim, x_dim, CV_8UC1);

	int num_triangles = trianglesA.size(); 
	for (int i = 0; i < num_triangles; i++) {
		current_maskA = Mat::zeros(y_dim, x_dim, CV_8UC1);
		current_maskB = Mat::zeros(y_dim, x_dim, CV_8UC1);
		set_mask_to_triangle(current_maskA, trianglesA[i]);
		set_mask_to_triangle(current_maskB, trianglesB[i]);
		Mat temp_imgA = Mat::zeros(y_dim, x_dim, CV_8UC1);
		Mat temp_imgB = Mat::zeros(y_dim, x_dim, CV_8UC1);
		Mat temp_imgC = Mat::zeros(y_dim, x_dim, CV_8UC1);
		imgA.copyTo(temp_imgA, current_maskA);
		imgB.copyTo(temp_imgB, current_maskB);
		warpAffine(temp_imgA, temp_imgA, get_affine_intermediate(affine_forward[i], tau), Size(x_dim, y_dim));
		warpAffine(temp_imgB, temp_imgB, affine_reverse[i], Size(x_dim, y_dim));
		warpAffine(temp_imgB, temp_imgB, get_affine_intermediate(affine_forward[i], tau), Size(x_dim, y_dim));
		addWeighted(temp_imgA, tau, temp_imgB, 1 - tau, 0.0, temp_imgC);
		addWeighted(temp_imgC, 1, canvas, 1, 0.0, canvas);
	}

	imshow("purple", canvas);
	waitKey(1);

	cout << "mesh test";
}

void interpolate_frame(MatchedGeometry g, string imgA_path, string imgB_path) {

	string root_path = "../data_store";
	Mat imgA = cv::imread(root_path + "/" + imgA_path, IMREAD_GRAYSCALE);
	Mat imgB = cv::imread(root_path + "/" + imgB_path, IMREAD_GRAYSCALE);
	Size desired_size = imgB.size();
	resize(imgA, imgA, desired_size);

	Rect img_rect = Rect(0, 0, desired_size.width, desired_size.height);

	vector<Vec6f> trianglesA = g.source_geometry.triangles;
	vector<Vec6f> trianglesB = g.target_geometry.triangles;

	// Forwards and reverse affine transformation parameters.
	vector<Mat> affine_forward = get_affine_transforms(trianglesA, trianglesB);
	vector<Mat> affine_reverse = get_affine_transforms(trianglesB, trianglesA);
	// should be a for loop for tau = 0 to tau = 1 with 0.01 jumps, but for now we will pick t = 0.6.
	float tau = 0.6;

	for (int i = 1; i < 10; i++) {
		std::clock_t start;
		double duration;
		start = clock();
		tau = (float)i * 0.1;
		cout << "Getting for tau: " << tau << endl;
		save_frame_at_tau(imgA, imgB, img_rect, affine_forward, affine_reverse, trianglesA, trianglesB, tau);
		duration = (clock() - start) / (double)CLOCKS_PER_MS;
		cout << "Amount of time for tau = " << tau << " : " << duration << endl;
	}
}


void write_mvv_header(char *version, int widthA, int heightA, int widthB, int heightB, int num_frames) {
	// header is 64 bytes

	int version_length = 4;

	std::ofstream ofile("../data_store/mvv_files/frame.mvv", std::ios::binary);
	
	for (int i = 0; i < version_length; i++) {
		ofile.write((char*)&version[i], sizeof(char));
	}
	ofile.write((char*)&widthA, sizeof(int));
	ofile.write((char*)&heightA, sizeof(int));
	ofile.write((char*)&widthB, sizeof(int));
	ofile.write((char*)&heightB, sizeof(int));
	ofile.write((char*)&num_frames, sizeof(int));

	int num_zeros = 10;
	// numBytes = 4*numZeros
	int zero = 0;
	for (int j = 0; j < num_zeros; j++) {
		ofile.write((char*)&zero, sizeof(int));
	}

	ofile.close();
}

void save_frame_master(string img1_path, string img2_path) {
	MatchedGeometry geometry = read_matched_points_from_file(img1_path, img2_path);

	vector<Vec6f> trianglesA = geometry.source_geometry.triangles;
	vector<Vec6f> trianglesB = geometry.target_geometry.triangles;
	
	Rect imgA_bounds = geometry.source_geometry.img;
	Rect imgB_bounds = geometry.target_geometry.img;

	vector<vector<Point>> rastered_trianglesA = raster_triangulation(trianglesA, imgA_bounds);
	vector<vector<Point>> rastered_trianglesB = raster_triangulation(trianglesB, imgB_bounds);

	int widthA = imgA_bounds.width;
	int heightA = imgA_bounds.height;
	int widthB = imgB_bounds.width;
	int heightB = imgB_bounds.height;

	// save affine params as .csv
	// save image raster as grayscale .png from 0-65536 (2 images)
	short** gridA = grid_from_raster(widthA, heightA, rastered_trianglesA);
	short** gridB = grid_from_raster(widthB, heightB, rastered_trianglesB);
	save_raster("../../data_store/raster/rasterA.bin", gridA, widthA, heightA);
	save_raster("../../data_store/raster/rasterB.bin", gridB, widthB, heightB);

	vector<Mat> affine_forward = get_affine_transforms(trianglesA, trianglesB);
	vector<Mat> affine_reverse = get_affine_transforms(trianglesB, trianglesA);

	float* affine_params = convert_vector_params(affine_forward, affine_reverse);
	write_float_array("../../data_store/affine/affine_1.bin", affine_params, trianglesA.size() * 12);

	cout << "Finished.";

	cin.get();
}

int danny_test() {
	// master function for constructing and saving a frame

	string img1_path = "david_1.jpg";
	string img2_path = "david_2.jpg";
	save_frame_master(img1_path, img2_path);

	//test_binary();

	return 0;
}

int adrian_test() {
	//vector<Point2f> corners;
	//trackbarCorners("..\\data_store\\david_1.jpg", corners);
	//test_match_points("david_1.jpg", "david_2.jpg");

	//test_match_points_2("david_1.jpg", "david_2.jpg");
	//test_GFTT("david_1.jpg", "david_2.jpg");
	//test_AGAST("david_1.jpg", "david_2.jpg");
	//test_BRISK("david_1.jpg", "david_2.jpg");
	//test_FAST("david_1.jpg", "david_2.jpg");
	//test_ORB("david_1.jpg", "david_2.jpg");
	//test_affine_ORB("david_1.jpg", "david_2.jpg");

	//test_kmeans("david_1.jpg", "david_2.jpg");
	//vector<KeyPoint> bs1;
	//vector<KeyPoint> bs2;
	//affine_akaze_test("..\\data_store\\david_1.jpg", "..\\data_store\\david_2.jpg", bs1, bs2);
	//test_nbh_first("david_1.jpg", "david_2.jpg");
	//test_nbh_grow("david_1.jpg", "david_2.jpg");
	//test_one_nbh("david_1.jpg", "david_2.jpg");
	//test_akaze_harris_global_harris_local("david_1.jpg", "david_2.jpg");

	//loading an david seems to take approx 50ms, so two davids 100ms
	string address1 = "..\\data_store\\david_1.jpg";
	Mat img1 = imread(address1, IMREAD_GRAYSCALE);
	string address2 = "..\\data_store\\david_2_resize_no_background.jpg";
	Mat img2 = imread(address2, IMREAD_GRAYSCALE);
	chrono::high_resolution_clock::time_point t1 = chrono::high_resolution_clock::now();


	chrono::high_resolution_clock::time_point t2 = chrono::high_resolution_clock::now();
	chrono::duration<double, std::milli> time_span = t2 - t1;
	std::cout << "It took me " << time_span.count() / 10 << " milliseconds." << endl;
	return 0;
}

int main()
{
	cout << APPLICATION_NAME << " version " << VERSION << endl;
	cout << COMPANY_NAME << " " << COPYRIGHT_YEAR << ". " << "All rights reserved." << endl;

	ifstream file("user_debug.txt");
	string str;
	getline(file, str);

	// Horrible hackish way of avoiding merge conflicts while we do testing

	if (str == "danny") {
		danny_test();
	}
	else if (str == "adrian") {
		adrian_test();
	}
	else {
		cout << "Invalid user";
	}


	cout << "Finished. Press enter twice to terminate program.";

	cin.get();

    return 0;
} 