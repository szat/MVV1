#include "mvv_iostream.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <chrono>
#include <vector>

using namespace std;
using namespace cv;

// what IO functions should we have:

float* convert_vector_params(vector<Mat> forward_params, vector<Mat> reverse_params) {
	int num_triangles = forward_params.size();
	float* params = new float[num_triangles*12];
	for (int i = 0; i < num_triangles; i++) {
		int inc = 12 * i;
		params[inc] = (float)forward_params[i].at<double>(0, 0);
		params[inc+1] = (float)forward_params[i].at<double>(0, 1);
		params[inc+2] = (float)forward_params[i].at<double>(0, 2);
		params[inc+3] = (float)forward_params[i].at<double>(1, 0);
		params[inc+4] = (float)forward_params[i].at<double>(1, 1);
		params[inc+5] = (float)forward_params[i].at<double>(1, 2);
		params[inc+6] = (float)reverse_params[i].at<double>(0, 0);
		params[inc+7] = (float)reverse_params[i].at<double>(0, 1);
		params[inc+8] = (float)reverse_params[i].at<double>(0, 2);
		params[inc+9] = (float)reverse_params[i].at<double>(1, 0);
		params[inc+10] = (float)reverse_params[i].at<double>(1, 1);
		params[inc+11] = (float)reverse_params[i].at<double>(1, 2);
	}
	return params;
}
