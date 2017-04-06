// mvv_trial.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <vector>

#include "fed.h"
#include "nldiffusion_functions.h"

using namespace cv;
using namespace std;
int main()
{
	vector<float> some_vec = { 1.2f, 3.2f, 4.4f };
	Mat src = imread("..\\data_store\\lena.bmp", CV_LOAD_IMAGE_COLOR);
	if (!src.data) {
		cout << "Could not open image" << endl;  
		return -1;
	}
	Mat dst;
	
	//testing the different headers
	cout << "Testing fed.h" << endl;
	cout << "fed_is_prime_internalV2(2) = " << fed_is_prime_internalV2(2) << endl;
	cout << "fed_tau_by_cycle_timeV2(2.3, 3.1, true, { 1.2, 3.2, 4.4 }) = " << fed_tau_by_cycle_timeV2(2.3, 3.1, true, some_vec) << endl;
	cout << "fed_tau_by_process_timeV2(2.3, 6, 3.1, true, { 1.2, 3.2, 4.4 }) = " << fed_tau_by_process_timeV2(2.3, 6, 3.1, true, some_vec) << endl;
	cout << "fed_tau_by_process_timeV2(2.3, 6, 3.1, true, { 1.2, 3.2, 4.4 }) = " << fed_tau_by_process_timeV2(2.3, 6, 3.1, true, some_vec) << endl;
	cout << "fed_tau_internalV2(4, 3.2, 3.5, true, { 1.2, 3.2, 4.4 }) = " << fed_tau_internalV2(4, 3.2, 3.5, true, some_vec) << endl;
	cout << endl;
	cout << "Testing nldiffusion_functions.h" << endl;
	cout << "gaussian_2D_convolutionV2(src, dst, 1, 1, 2.1f) about to start..." << endl;
	gaussian_2D_convolutionV2(src, dst, 1, 1, 2.1f);
	cout << "gaussian_2D_convolutionV2(src, dst, 1, 1, 2.1f) about to finished!" << endl;

	Size srcSize = src.size();
	Size dstSize = dst.size();

	Mat im3(sz1.height, sz1.width + sz2.width, CV_8UC3);
	Mat left(im3, Rect(0, 0, sz1.width, sz1.height));
	im1.copyTo(left);
	Mat right(im3, Rect(sz1.width, 0, sz2.width, sz2.height));
	im2.copyTo(right);
	imshow("im3", im3);
	waitKey(0);
	return 0;
	cin.ignore();
	return 0;
}
