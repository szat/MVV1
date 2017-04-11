// harris_corners_opencv.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
using namespace cv;
using namespace std;

int main(int, char** argv)
{
	int thresh = 200;
	Mat src, gray;
	// Load source image and convert it to gray
	src = imread("..\\data_store\\mona_lisa_1.jpg", 1);
	cvtColor(src, gray, CV_BGR2GRAY);
	Mat dst, dst_norm, dst_norm_scaled;
	dst = Mat::zeros(src.size(), CV_32FC1);

	// Detecting corners
	cornerHarris(gray, dst, 20, 9, 0.05, BORDER_DEFAULT);

	// Normalizing
	normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	convertScaleAbs(dst_norm, dst_norm_scaled);

	// Drawing a circle around corners
	for (int j = 0; j < dst_norm.rows; j++)
	{
		for (int i = 0; i < dst_norm.cols; i++)
		{
			if ((int)dst_norm.at<float>(j, i) > thresh)
			{
				circle(dst_norm_scaled, Point(i, j), 5, Scalar(0), 2, 8, 0);
			}
		}
	}


	// Showing the result
	namedWindow("corners_window", CV_WINDOW_AUTOSIZE);
	imshow("corners_window", dst_norm_scaled);

	waitKey(0);

	return(0);
}
