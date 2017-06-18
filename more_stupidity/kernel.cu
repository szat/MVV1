
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <string>
#include <stdio.h>
#include <iostream>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/utility.hpp>

//#include <opencv2/features2d/features2d.hpp>
#include <AKAZE.h>
#include <AKAZEConfig.h>
#include <opencv2/calib3d.hpp> //AKAZE seems not to work without this

int main() {
	cv::Mat img1;
	std::string img1_path = "..\\data_store\\images\\c1_img_000177.png";
	img1 = cv::imread(img1_path);
	if (img1.empty())
	{
		std::cout << "Could not open image..." << img1_path << "\n";
		return -1;
	}

	AKAZEOptions options;
	libAKAZECU::AKAZE evolution1(options);

	return 0;
}