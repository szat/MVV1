// spx_stats.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/ximgproc.hpp>
#include <opencv2/bgsegm.hpp>
#include <opencv2/videoio.hpp>


#include <stdio.h>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>

using namespace std;
using namespace cv;
using namespace cv::ximgproc;
using namespace cv::bgsegm;

//TODO check for maximal number of superpixels

class SuperPx {
public:
	int label;
	int size;

	double length;      // Length of a box
	double breadth;     // Breadth of a box
	double height;      // Height of a box

	double getVolume(void) {
		return length * breadth * height;
	}
};

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
				crushed.at<int>(i, j) = -1;
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

vector<Point2d> get_one_contour(Mat & crushed, int label) {
	vector<Point2d> out;
	for (int i = 0; i < crushed.rows; i++) {
		for (int j = 0; j < crushed.cols; j++) {
			if (crushed.at<int>(i, j) == label) {
				Point2d pt(i, j);
				out.push_back(pt);
			}
		}
	}
	return out;
}

float get_variance(Mat & image, Mat & labels, int label) {
	float variance = 0;
	int n = 0;
	float mean = 0;

	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			if (labels.at<int>(i, j) == 0) {
				float x = image.at<Vec3b>(Point(i, j))[2];
				n++;
				float delta = x - mean;
				mean += delta / n;
				float delta2 = x - mean;
				variance += delta*delta2;
			}
		}
	}
	if(n >= 2)	return variance / (n - 1);
	else return 0;
}

float get_features(Mat & image, Mat & labels, int number_of_spx, int number_of_features) {
	float mean = 0;
	int count = 0;
	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			if (labels.at<int>(i, j) == 0) {
				count++;
				mean = mean + image.at<Vec3b>(Point(i, j))[2];
			}
		}
	}
	return mean / count;
	/*
	//initialization
	vector<vector<float>> spx_features;
	for (int i = 0; i < number_of_spx; i++) {
		vector<float> features;
		for (int j = 0; j < number_of_features; j++) {
			features.push_back(0);
		}
		spx_features.push_back(features);
	}

	//So each spx has a feature vector of sie number_of_features
	vector<float> spx_size;
	for (int i = 0; i < number_of_spx; i++) {
		spx_size.push_back(0);
	}

	Mat temp = image.clone(); 
	//cvtColor(image, temp, CV_8UC3);
	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			spx_size.at(labels.at<int>(i, j))++;
			Vec3b color = temp.at<Vec3b>(Point(i, j));
			spx_features.at(labels.at<int>(i, j)).at(0) += (int)color[0];
			spx_features.at(labels.at<int>(i, j)).at(1) += (int)color[1];
			spx_features.at(labels.at<int>(i, j)).at(2) += (int)color[2]; 
		}
	}

	for (int i = 0; i < number_of_spx; i++) {
		spx_features.at(i).at(0) = spx_features.at(i).at(0) / spx_size.at(i);
		spx_features.at(i).at(1) = spx_features.at(i).at(1) / spx_size.at(i);
		spx_features.at(i).at(2) = spx_features.at(i).at(2) / spx_size.at(i);
	}

	return spx_features;	
	*/
}

vector<vector<Point2d>> get_contour(Mat & crushed, vector<Point2d> & starting_pts) {
	vector<vector<Point2d>> out;
	for (int i = 0; i < starting_pts.size(); i++) {
		vector<Point2d> contour;
		out.push_back(contour);
	}

	for (int i = 0; i < crushed.rows; i++) {
		for (int j = 0; j < crushed.cols; j++) {
			if (crushed.at<int>(i, j) != -1) {
				Point2d pt(i, j);
				out.at((int)crushed.at<int>(i, j)).push_back(pt);
			}
		}
	}

	return out;
}

static Ptr<BackgroundSubtractor> createBGSubtractorByName(const String& algoName)
{
	Ptr<BackgroundSubtractor> algo;
	if (algoName == String("GMG"))
		algo = createBackgroundSubtractorGMG(20, 0.7);
	else if (algoName == String("CNT"))
		algo = createBackgroundSubtractorCNT();
	else if (algoName == String("KNN"))
		algo = createBackgroundSubtractorKNN();
	else if (algoName == String("MOG"))
		algo = createBackgroundSubtractorMOG();
	else if (algoName == String("MOG2"))
		algo = createBackgroundSubtractorMOG2();

	return algo;
}

int main()
{
	/*
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

	auto t1 = std::chrono::high_resolution_clock::now();
	vector<vector<Point2d>> contours = get_contour(crushed, starting_pts);
	auto t2 = std::chrono::high_resolution_clock::now();
	std::cout << "Total: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "ms" << endl;

	cout << "Computation done!" << endl;

	int number_of_spx = slic->getNumberOfSuperpixels();
	float spx_features = get_features(img, labels, number_of_spx, 3);
	float spx_varaince = get_variance(img, labels, 0);
	*/
	
	// Playing with background segmentation

	std::cout << "Using OpenCV version " << CV_VERSION << "\n" << std::endl;
	std::cout << getBuildInformation();

	VideoCapture capture("C:/Users/Adrian/Desktop/MVI_5482.avi");
	if (!capture.isOpened()) {
		std::cout << "cannot read video!\n";
		return -1;
	}

	Mat frame;
	//namedWindow("frame");
	namedWindow("foreground");

	double rate = capture.get(CV_CAP_PROP_FPS);
	int delay = 1000 / rate;

	Ptr<BackgroundSubtractor> bgfs = createBGSubtractorByName("MOG");	
	//KNN not that bad, MOG2 better, seems fast
	//"GMG","CNT","CNT","KNN","MOG","MOG","MOG2"
	Mat fgmask, segm;

	while (true)
	{
		if (!capture.read(frame)) {
			break;
		}
		//imshow("frame", frame);

		bgfs->apply(frame, fgmask);
		frame.convertTo(segm, CV_8U, 0.5);
		add(frame, Scalar(100, 100, 0), segm, fgmask);

		imshow("foreground", segm);

		int c = waitKey(20);
		if (c == 'q' || c == 'Q' || (c & 255) == 27)
			break;
	}
	
    return 0;
}

