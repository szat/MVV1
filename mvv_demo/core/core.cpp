// core.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <vector>
#include <ctime>
#include <stdio.h>

#include "build_geometry.h"
#include "generate_test_points.h"

#include "match_points.h"
#include "good_features.h"
#include "affine_akaze.h"

#define VERSION "1.0.0"
#define APPLICATION_NAME "MVV"
#define COMPANY_NAME "NDim Inc."
#define COPYRIGHT_YEAR 2017

using namespace std;
using namespace cv;

int corner_points_test() {
	cout << "Begin integration test of match_features and build_geometry" << endl;

	//this is the image used in trackbarCorners
	Mat src1 = imread("..\\data_store\\david_1.jpg", IMREAD_GRAYSCALE); 
	vector<Point2f> corners;
	trackbarCorners(corners);


	Rect testRect = Rect(0, 0, src1.size().width, src1.size().height);
	graphical_triangulation(corners, testRect);

	return 0;
}

int main()
{
	cout << APPLICATION_NAME << " version " << VERSION << endl;
	cout << COMPANY_NAME << " " << COPYRIGHT_YEAR << ". " << "All rights reserved." << endl;

	// Danny current test

	corner_points_test();

	//vector<Vec6f> triangleSet1 = test_interface();

	// Adrian current test
	//affine_akaze();
	//test_match_points();
	//test_trackbar2(0);

	cout << "Finished. Press enter twice to terminate program.";
	cin.get();

    return 0;
}

