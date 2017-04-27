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
	Subdiv2D subdiv = raw_triangulation(imagePointsA, imageSizeA);
	display_triangulation(subdiv, imageSizeA);


	return 0;
}

void image_diagnostics(Mat img) {
	Size size = img.size();
	cout << "Image width: " << size.width << endl;
	cout << "Image height: " << size.height << endl;
}

int triangulation_diagnostic() {
	// This function takes two input images from two angles (rescaled to the same size)
	// There will be a slider that controls the parameter t (0,1), so we can see the discrete triangulation shifts
	string imagePathA = "david_1.jpg";
	string imagePathB = "david_2.jpg";
	string rootPath = "../data_store";

	//Rect imageSizeA = getImageSize(imageA);
	//Rect imageSizeB = getImageSize(imageB);

	Mat imgA = cv::imread(rootPath + "/" + imagePathA, IMREAD_GRAYSCALE);
	Mat imgB = cv::imread(rootPath + "/" + imagePathB, IMREAD_GRAYSCALE);

	// Resize imageB so that is has the same size as imgA
	cv::resize(imgB, imgB, imgA.size());

	Rect imgSizeRectA = Rect(0, 0, imgA.size().width, imgA.size().height);
	vector<vector<KeyPoint>> pointMatches = match_points_mat(imgA, imgB);
	
	vector<KeyPoint> imgPointsA = pointMatches[0];
	vector<KeyPoint> imgPointsB = pointMatches[1];

	triangulation_trackbar(imgPointsA, imgPointsB, imgSizeRectA);

	return -1;
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
};

MatchedGeometry create_matched_geometry(vector<Point2f> imgPointsA, vector<Point2f> imgPointsB, Rect imgSizeRectA, Rect imgSizeRectB) {
	if (imgPointsA.size() != imgPointsB.size()) {
		throw "Matched points must have the same size (imgPointsA.size != imgPointsB.size)";
	}

	// triangulate source interior
	vector<Vec6f> trianglesA = construct_triangles(imgPointsA, imgSizeRectA);

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
	vector<pair<Vec4f, Vec4f>> trapezoidsA = project_trapezoids_from_hull(convexHullA, imgSizeRectA, centerOfMassA);
	vector<pair<Vec4f, Vec4f>> trapezoidsB = project_trapezoids_from_hull(convexHullB, imgSizeRectB, centerOfMassB);

	// calculate priority (triangles)
	vector<int> interiorPriority = calculate_triangle_priority(trianglesB);
	// calculate priority (trapezoids)
	vector<int> exteriorPriority = calculate_trapezoid_priority(trapezoidsB);

	// This could potentially be replaced by two constructors.
	MatchedGeometry matchedResult = MatchedGeometry();
	GeometricSlice source = GeometricSlice();
	GeometricSlice target = GeometricSlice();
	source.img = imgSizeRectA;
	source.triangles = trianglesA;
	source.trapezoids = trapezoidsA;
	target.img = imgSizeRectB;
	target.triangles = trianglesB;
	target.trapezoids = trapezoidsB;
	matchedResult.sourceGeometry = source;
	matchedResult.targetGeometry = target;
	matchedResult.trianglePriority = interiorPriority;
	matchedResult.trapezoidPriority = exteriorPriority;
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

	Rect imgSizeRectA = Rect(0, 0, imgA.size().width, imgA.size().height);
	Rect imgSizeRectB = Rect(0, 0, imgB.size().width, imgB.size().height);

	vector<vector<KeyPoint>> pointMatches = match_points_mat(imgA, imgB);

	vector<KeyPoint> imgKeyPointsA = pointMatches[0];
	vector<KeyPoint> imgKeyPointsB = pointMatches[1];
	vector<Point2f> imgPointsA = convert_key_points(imgKeyPointsA);
	vector<Point2f> imgPointsB = convert_key_points(imgKeyPointsB);

	MatchedGeometry geometry = create_matched_geometry(imgPointsA, imgPointsB, imgSizeRectA, imgSizeRectB);
	return geometry;
}


int test_5_points() {
	Rect imgRectA = Rect(0, 0, 500, 600);
	Rect imgRectB = Rect(0, 0, 500, 600);

	vector<Point2f> pointsA = vector<Point2f>();
	vector<Point2f> pointsB = vector<Point2f>();

	pointsA.push_back(Point2f(200, 300));
	pointsA.push_back(Point2f(400, 300));
	pointsA.push_back(Point2f(300, 500));

	pointsB.push_back(Point2f(190.213, 313.219));
	pointsB.push_back(Point2f(412.092, 290.012));
	pointsB.push_back(Point2f(329, 523.3234));

	MatchedGeometry m = create_matched_geometry(pointsA, pointsB, imgRectA, imgRectB);
	return -1;
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

int interpolation_preprocessing() {
	// 
	return -5;
}

void render_frame_t() {

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

int test_interpolation() {
	// in interpolation preprocessing, we must:
	// calculate the A,B matrices (i.e. the ) for EACH set of triangles

	// in render_frame_t, we (at the beginning) calculate the A(t), B(t) parametrized matrices for each triangle at this step t
	// render triangle-by-triangle

	// assuming we have affine transformations that are parametrized, how exactly do we mesh the images together?????

	// we gonna try this with 1 triangle

	Vec6f tri1 = Vec6f(0, 0, 100, 0, 100, 100);
	Vec6f tri2 = Vec6f(50, 0, 150, 40, 155, 250);

	Mat aff = affine_transform(tri1, tri2);
	Size size = aff.size();

	double a00 = aff.at<double>(0, 0);
	double a01 = aff.at<double>(0, 1);
	double b00 = aff.at<double>(0, 2);
	double a10 = aff.at<double>(1, 0);
	double a11 = aff.at<double>(1, 1);
	double b01 = aff.at<double>(1, 2);

	vector<Point2f> tri1points = vector<Point2f>();
	tri1points.push_back(Point2f(0, 0));
	tri1points.push_back(Point2f(100, 0));
	tri1points.push_back(Point2f(100, 100));

	int points = 3;
	vector<Point2f> points1 = parametrized_interpolation(0, tri1points, a00, a01, a10, a11, b00, b01);
	vector<Point2f> points2 = parametrized_interpolation(0.5, tri1points, a00, a01, a10, a11, b00, b01);
	vector<Point2f> points3 = parametrized_interpolation(1, tri1points, a00, a01, a10, a11, b00, b01);

	return -1;
}

int danny_test() {
	//test_5_points();
	//MatchedGeometry geometry = read_matched_points_from_file("david_1.jpg", "david_2.jpg");
	//render_matched_geometry(geometry.sourceGeometry, "Test window 1");
	//render_matched_geometry(geometry.targetGeometry, "Test window 2");

	//Vec6f testTri = Vec6f(0, 0, 100, 100, 0, 100);
	//float result = get_triangle_area(testTri);

	test_interpolation();
	

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
	test_nbh_first("david_1.jpg", "david_2.jpg");
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