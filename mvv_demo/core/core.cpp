// core.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <vector>
#include <ctime>
#include <stdio.h>

#include "construct_geometry.h"
#include "generate_test_points.h"

#define VERSION "1.0.0"
#define APPLICATION_NAME "MVV"
#define COMPANY_NAME "NDim Inc."
#define COPYRIGHT_YEAR 2017

using namespace std;
using namespace cv;

/*
std::vector<cv::Point2f> get_sample_points() {
	std::vector<cv::Point2f> thing = std::vector<cv::Point2f>();
	return thing;
}
*/

int main()
{

	cout << APPLICATION_NAME << " version " << VERSION << endl;
	cout << COMPANY_NAME << " " << COPYRIGHT_YEAR << ". " << "All rights reserved." << endl;
	cout << "Beginning startup process..." << endl;

	std::vector<cv::Point2f> samplePoints;
	//= get_sample_points();
	//samplePoints = get_sample_points();
	
	std::vector<cv::Point2f> thingy = get_sample_points();



	Rect sampleRect = Rect(0, 0, 600, 600);
	//vector<Vec6f> triangles = construct_triangles(samplePoints, sampleRect);
	

	cin.get();

    return 0;
}

