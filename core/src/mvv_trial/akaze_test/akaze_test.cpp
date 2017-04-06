// mvv_trial.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
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
	cout << "To see the effect of a filter it is easiest to use ImageWatch." << endl;
	cout << endl;

	cout << "gaussian_2D_convolutionV2(src, dst, 1, 1, 2.1f)" << endl;
	gaussian_2D_convolutionV2(src, dst, 1, 1, 2.1f); 
	//another way to view the effect of a filter is to put breakpoints and to use ImageWatch
	/*
	namedWindow("src", WINDOW_AUTOSIZE);
	imshow("src", src);
	namedWindow("dst", WINDOW_AUTOSIZE);
	imshow("dst", dst);
	waitKey(0); //necessary with for namedWindow and imshow to work
	destroyAllWindows();
	*/
	cout << endl;

	cout << "image_derivatives_scharrV2(src, dst, dx=0, dy=1)" << endl;
	cout << "dx and dy have to be either 0 or 1 and sum to one, otherwise (dy >= 0 && dx >= 0 && dx+dy == 1) fails." << endl;
	image_derivatives_scharrV2(src, dst, 0, 1);
	cout << endl;

	cout << "Construct Mat srcDx with image_derivatives_scharrV2(src, srcDx, 1, 0)" << endl;
	Mat srcDx; 
	image_derivatives_scharrV2(src, srcDx, 1, 0);
	Mat grayDx(srcDx.size(), CV_32F);
	cvtColor(srcDx, grayDx, CV_BGR2GRAY);
	cout << "Construct Mat srcDy with image_derivatives_scharrV2(src, srcDy, 0, 1)" << endl;
	Mat srcDy;
	image_derivatives_scharrV2(src, srcDy, 0, 1);
	Mat grayDy(srcDy.size(), CV_32F);
	cvtColor(srcDy, grayDy, CV_BGR2GRAY);
	cout << endl;

	cout << "Compute diffusion coefs, safer to have both input and output matrices grayscale." << endl;
	
	cout << "pm_g1V2(srcDx, srcDy, dst, 20.2f)" << endl;
	Mat grayDst1(src.size(), CV_32F);
	pm_g1V2(grayDx, grayDy, grayDst1, 20.2f);
	
	cout << "pm_g2V2(srcDx, srcDy, dst, 20.2f)" << endl;
	Mat grayDst2(src.size(), CV_32F);
	pm_g2V2(grayDx, grayDy, grayDst2, 20.2f);
	
	cout << "weickert_diffusivityV2(grayDx, grayDy, grayDst3, 20.2f)" << endl;
	Mat grayDst3(src.size(), CV_32F);
	weickert_diffusivityV2(grayDx, grayDy, grayDst3, 20.2f);


	cin.ignore();

	return 0;
}
