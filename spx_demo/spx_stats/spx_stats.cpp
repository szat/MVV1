// spx_stats.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/ximgproc.hpp>

#include <stdio.h>
#include <iostream>
#include <string>
#include <vector>

using namespace std;
using namespace cv;
using namespace cv::ximgproc;

//TODO check for maximal number of superpixels

Mat visualize_labels(Mat labels) {
	Mat label_viz(labels.size(), CV_8UC3);

	int width = labels.size().width;
	for (int i = 0; i < labels.rows; i++) {
		for (int j = 0; j < labels.cols; j++) {
			label_viz.at<Vec3b>(i, j)[0] = labels.at<int>(i, j) % 255;
			label_viz.at<Vec3b>(i, j)[1] = labels.at<int>(i, j) - labels.at<int>(i, j) % 255;
			label_viz.at<Vec3b>(i, j)[1] = labels.at<int>(i, j) / 2 % 255;
		}
	}
	return label_viz;
}

Mat crush_to_contour(Mat & labels) {
	Mat crushed = labels.clone();
	int width = labels.size().width;

	//the boundary ofthe matrix is part of the contour by default
	for (int i = 1; i < labels.rows-1; i++) {
		for (int j = 1; j < labels.cols-1; j++) {
			int up = labels.at<int>(i - 1, j);
			int down = labels.at<int>(i + 1,j);
			int left = labels.at<int>(i, j - 1);
			int right = labels.at<int>(i, j + 1);

			int current = labels.at<int>(i, j);
			if (current == up && current == down && current == left && current == right) {
				crushed.at<int>(i, j) = 0;
			}
		}
	}
	return crushed;
}

vector<Point2d> get_contour_starting_pts(Mat & labels) {
	vector<Point2d> out;
	int counter = 0;
	
	for (int i = 0; i < labels.rows; i++) {
		for (int j = 0; j < labels.cols; j++) {
			if (labels.at<int>(i, j) == counter) {
				Point2d start(i, j);
				out.push_back(start);
				counter++;
			}
		}
	}

	return out;
}

vector<Point2d> get_contour(Mat & labels, int label) {
	//check for maximal number of segments
	vector<Point2d> out;
	//find first instance of the label
	return out;
}

int main()
{
	cout << "Welcome to spx_demo, code to try out the SLIC segmentation method!" << endl;
	cout << "Press any key to exit." << endl;

	string img_file = "..\\..\\data_store\\images\\david_1.jpg";
	Mat img;

	img = imread(img_file);
	if (img.empty())
	{
		cout << "Could not open image..." << img_file << "\n";
		return -1;
	}

	Mat converted;
	cvtColor(img, converted, COLOR_BGR2HSV);

	int algorithm = 0;
	int region_size = 25;
	int ruler = 45;
	int min_element_size = 50;
	int num_iterations = 5;

	cout << "New computation!" << endl;

	Ptr<SuperpixelSLIC> slic = createSuperpixelSLIC(converted, algorithm + SLIC, region_size, float(ruler));
	slic->iterate(num_iterations);
	if (min_element_size > 0) 	slic->enforceLabelConnectivity(min_element_size);
	
	Mat result, mask;
	result = img.clone();
	slic->getLabelContourMask(mask, true);

	result.setTo(Scalar(0, 255, 0), mask);

	Mat labels;
	slic->getLabels(labels);

	Mat label_viz = visualize_labels(labels);

	Mat crushed = crush_to_contour(labels);

	vector<Point2d> starting_pts = get_contour_starting_pts(labels);

	cout << "The length of starting_pts is " << starting_pts.size() << endl;
	cout << "The last one should start at " << starting_pts.at(starting_pts.size() - 1).x << " , " << starting_pts.at(starting_pts.size() - 1).y << endl;

	Mat crushed_viz = visualize_labels(crushed);

	cout << "Computation done!" << endl;

    return 0;
}

