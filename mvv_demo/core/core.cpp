// core.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <vector>
#include <ctime>
#include <stdio.h>
#include <stdlib.h>

#include "build_geometry.h"
#include "generate_test_points.h"

#include "match_points.h"
#include "good_features.h"
#include "affine_akaze.h"
#include "knn_test.h"

#include "interpolate_images.h"

#define VERSION "1.0.0"
#define APPLICATION_NAME "MVV"
#define COMPANY_NAME "NDim Inc."
#define COPYRIGHT_YEAR 2017
#define CLOCKS_PER_MS (CLOCKS_PER_SEC / 1000)

using namespace std;
using namespace cv;

int corner_points_test() {
	cout << "Begin integration test of match_features and build_geometry" << endl;

	//this is the image used in trackbarCorners
	string imagePath = "..\\data_store\\david_1.jpg";
	Mat src1 = imread(imagePath, IMREAD_GRAYSCALE); 
	vector<Point2f> corners;
	trackbarCorners(imagePath, corners);


	Rect testRect = Rect(0, 0, src1.size().width, src1.size().height);
	graphical_triangulation(corners, testRect);

	return 0;
}

Rect getImageSize(string imagePath) {
	string address = "..\\data_store\\" + imagePath;
	string input = "";
	ifstream infile1;
	infile1.open(address.c_str());

	Mat img1 = imread(address, IMREAD_GRAYSCALE);

	float height = img1.size().height;
	float width = img1.size().width;

	Rect imageSize = Rect(0, 0, width, height);
	return imageSize;
}

int test_matching() {

	string imageA = "david_1.jpg";
	string imageB = "david_2.jpg";

	Rect imageSizeA = getImageSize(imageA);
	Rect imageSizeB = getImageSize(imageB);

	vector<vector<KeyPoint>> pointMatches = test_match_points(imageA, imageB);

	vector<KeyPoint> imageMatchesA = pointMatches.at(0);
	vector<KeyPoint> imageMatchesB = pointMatches.at(1);

	vector<Point2f> imagePointsA = vector<Point2f>();

	int vecSize = imageMatchesA.size();

	for (int i = 0; i < vecSize; i++) {
		imagePointsA.push_back(imageMatchesA.at(i).pt);
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
	vector<pair<Vec4f, Vec4f>> trapezoids;
};

struct MatchedGeometry {
	GeometricSlice sourceGeometry;
	GeometricSlice targetGeometry;
	vector<int> trianglePriority;
	vector<int> trapezoidPriority;
	Size imageSize;
};

MatchedGeometry create_matched_geometry(vector<Point2f> imgPointsA, vector<Point2f> imgPointsB, Size size) {
	// triangulate source interior
	vector<Vec6f> trianglesA = construct_triangles(imgPointsA, size);

	// triangulate target interior
	vector<Vec6f> trianglesB = triangulate_target(imgPointsA, imgPointsB, trianglesA);

	// Need a function to render triangles output
	// detect edges of source (convex hull)
	vector<int> convexHullIndices = get_source_convex_hull(imgPointsA);
	vector<Point2f> convexHullA = hull_indices_to_points(convexHullIndices, imgPointsA);
	vector<Point2f> convexHullB = hull_indices_to_points(convexHullIndices, imgPointsB);

	Point2f centerOfMassA = get_center_of_mass(imgPointsA);
	Point2f centerOfMassB = get_center_of_mass(imgPointsB);

	// construct target and source trapezoids  
	// use the same Key/Value mapping from triangulate_target
	Rect imgBounds = Rect(0, 0, size.width, size.height);
	vector<pair<Vec4f, Vec4f>> trapezoidsA = project_trapezoids_from_hull(convexHullA, imgBounds, centerOfMassA);
	vector<pair<Vec4f, Vec4f>> trapezoidsB = project_trapezoids_from_hull(convexHullB, imgBounds, centerOfMassB);

	// calculate priority (triangles)
	vector<int> interiorPriority = calculate_triangle_priority(trianglesB);
	// calculate priority (trapezoids)
	vector<int> exteriorPriority = calculate_trapezoid_priority(trapezoidsB);

	// This could potentially be replaced by two constructors.
	MatchedGeometry matchedResult = MatchedGeometry();
	GeometricSlice source = GeometricSlice();
	GeometricSlice target = GeometricSlice();
	source.triangles = trianglesA;
	source.trapezoids = trapezoidsA;
	target.triangles = trianglesB;
	target.trapezoids = trapezoidsB;
	matchedResult.sourceGeometry = source;
	matchedResult.targetGeometry = target;
	matchedResult.trianglePriority = interiorPriority;
	matchedResult.trapezoidPriority = exteriorPriority;
	matchedResult.imageSize = size;
	return matchedResult;
}

MatchedGeometry read_matched_points_from_file(string sourcePath, string targetPath) {
	// Please note that:
	// A: source image
	// B: target image
	// I use this shorthand so that the variable names are shorter.

	cout << "Initializing matched geometry routine" << endl;

	string imagePathA = sourcePath;
	string imagePathB = targetPath;
	string rootPath = "../data_store";
	Mat imgA = cv::imread(rootPath + "/" + imagePathA, IMREAD_GRAYSCALE);
	Mat imgB = cv::imread(rootPath + "/" + imagePathB, IMREAD_GRAYSCALE);

	Size desiredSize = imgB.size();
	resize(imgA, imgA, desiredSize);

	vector<vector<KeyPoint>> pointMatches = match_points_mat(imgA, imgB);

	vector<KeyPoint> imgKeyPointsA = pointMatches[0];
	vector<KeyPoint> imgKeyPointsB = pointMatches[1];
	vector<Point2f> imgPointsA = convert_key_points(imgKeyPointsA);
	vector<Point2f> imgPointsB = convert_key_points(imgKeyPointsB);

	MatchedGeometry geometry = create_matched_geometry(imgPointsA, imgPointsB, desiredSize);
	return geometry;
}

void render_matched_geometry(GeometricSlice slice, string windowName) {
	// Render both images.

	// Trapezoids in red (first)
	Mat img(slice.img.size(), CV_8UC3);
	Scalar trapezoid_color(0, 0, 255), triangle_color(255, 255, 255), hull_color(255, 0, 0);
	string win = windowName;

	vector<Point> pt(3);
	int numTriangles = slice.triangles.size();
	for (int i = 0; i < numTriangles; i++) {
		Vec6f t = slice.triangles[i];
		pt[0] = Point(cvRound(t[0]), cvRound(t[1]));
		pt[1] = Point(cvRound(t[2]), cvRound(t[3]));
		pt[2] = Point(cvRound(t[4]), cvRound(t[5]));
		line(img, pt[0], pt[1], triangle_color, 1, LINE_AA, 0);
		line(img, pt[1], pt[2], triangle_color, 1, LINE_AA, 0);
		line(img, pt[2], pt[0], triangle_color, 1, LINE_AA, 0);
	}

	vector<Point> ptr(4);
	int numTrapezoids = slice.trapezoids.size();
	for (int i = 0; i < numTrapezoids; i++) {
		Vec4f xCoords = slice.trapezoids[i].first;
		Vec4f yCoords = slice.trapezoids[i].second;
		ptr[0] = Point(cvRound(xCoords[0]), cvRound(yCoords[0]));
		ptr[1] = Point(cvRound(xCoords[1]), cvRound(yCoords[1]));
		ptr[2] = Point(cvRound(xCoords[2]), cvRound(yCoords[2]));
		ptr[3] = Point(cvRound(xCoords[3]), cvRound(yCoords[3]));
		line(img, ptr[0], ptr[1], trapezoid_color, 1, LINE_AA, 0);
		line(img, ptr[1], ptr[2], trapezoid_color, 1, LINE_AA, 0);
		line(img, ptr[2], ptr[3], trapezoid_color, 1, LINE_AA, 0);
		line(img, ptr[3], ptr[0], trapezoid_color, 1, LINE_AA, 0);
	}

	imshow(win, img);
	waitKey(1);
	// Triangles in white (second)
	// Border (convex hull in blue) (last)

}

vector<Point2f> parametrized_interpolation(float t, vector<Point2f> points, double a00, double a01, double a10, double a11, double b00, double b01) {
	vector<Point2f> paramPoints = vector<Point2f>();
	for (int i = 0; i < 3; i++) {
		float xP = (1 - t + a00 * t) * points[i].x + (a01 * t) * points[i].y + (b00 * t);
		float yP = (a10 * t) * points[i].x + (1 - t + a11 * t) * points[i].y + (b01 * t);
		paramPoints.push_back(Point2f(xP, yP));
	}
	return paramPoints;
}

int interpolate(MatchedGeometry g) {
	/*
	vector<vector<vector<double>>> affine = interpolation_preprocessing(g.sourceGeometry.triangles, g.targetGeometry.triangles);
	interpolation_trackbar(g.sourceGeometry.triangles, g.targetGeometry.triangles, g.sourceGeometry.img, g.targetGeometry.img, affine);
	*/
	return -1;
}

void set_mask_to_triangle(Mat &mask, Vec6f t) {
	Point pts[3] = {
		Point(t[0],t[1]),
		Point(t[2],t[3]),
		Point(t[4],t[5]),
	};
	fillConvexPoly(mask, pts, 3, Scalar(1));
}

void interpolate_frame(MatchedGeometry g, string imagePathA, string imagePathB) {
	std::clock_t start;
	double duration;
	start = clock();
	
	string rootPath = "../data_store";
	Mat imgA = cv::imread(rootPath + "/" + imagePathA, IMREAD_GRAYSCALE);
	Mat imgB = cv::imread(rootPath + "/" + imagePathB, IMREAD_GRAYSCALE);
	Size desiredSize = imgB.size();
	resize(imgA, imgA, desiredSize);

	Rect imgRect = Rect(0, 0, desiredSize.width, desiredSize.height);

	vector<Vec6f> trianglesA = g.sourceGeometry.triangles;
	vector<Vec6f> trianglesB = g.targetGeometry.triangles;

	// Forwards and reverse affine transformation parameters.
	vector<Mat> affineForward = get_affine_transforms(trianglesA, trianglesB);
	vector<Mat> affineReverse = get_affine_transforms(trianglesB, trianglesA);
	duration = (clock() - start) / (double)CLOCKS_PER_MS;
	// should be a for loop for tau = 0 to tau = 1 with 0.01 jumps, but for now we will pick t = 0.6.
	float tau = 0.6;

	for (int i = 1; i < 10; i++) {
		tau = (float)i * 0.1;
		cout << "Getting for tau: " << tau << endl;
		save_frame_at_tau(imgA, imgB, imgRect, affineForward, affineReverse, trianglesA, trianglesB, tau);
	}

	cout << "Amount of time for affine params: " << duration << endl;
}

int danny_test() {
	string img1path = "david_1.jpg";
	string img2path = "david_2.jpg";
	MatchedGeometry geometry = read_matched_points_from_file(img1path, img2path);
	interpolate_frame(geometry, img1path, img2path);
	//purple_mesh_test();
	return 0;
}

int adrian_test() {
	vector<Point2f> corners;
	trackbarCorners("..\\data_store\\david_1.jpg", corners);
	//test_match_points("david_1.jpg", "david_2.jpg");

	//test_match_points_2("david_1.jpg", "david_2.jpg");
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