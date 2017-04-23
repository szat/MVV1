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
	vector<pair<Vec4f, Vec4f>> trapezoidSource;
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
	// calculate priority (trapezoids)
	// return MatchedGeometry

	MatchedGeometry matchedResult = MatchedGeometry();
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

<<<<<<< HEAD
	// construct target and source trapezoids  
	// use the same Key/Value mapping from triangulate_target
	//vector<pair<Vec4f, Vec4f>> trapezoidsA = project_trapezoids_from_hull(convexHullA, imgSizeRectA);
	//vector<pair<Vec4f, Vec4f>> trapezoidsB = project_trapezoids_from_hull(convexHullB, imgSizeRectB);

	// calculate priority (triangles)
	// calculate priority (trapezoids)
	// return MatchedGeometry
=======
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
>>>>>>> 39759f4230d956b07ff48e450e362c220edbceeb

	MatchedGeometry m = create_matched_geometry(pointsA, pointsB, imgRectA, imgRectB);
	return -1;
}

int main()
{
	cout << APPLICATION_NAME << " version " << VERSION << endl;
	cout << COMPANY_NAME << " " << COPYRIGHT_YEAR << ". " << "All rights reserved." << endl;

	cout << "Finished. Press enter twice to terminate program.";
	cin.get();

    return 0;
} 