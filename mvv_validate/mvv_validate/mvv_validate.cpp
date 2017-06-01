// mvv_validate.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
//#include "binary_write.h"
//#include "binary_read.h"

using namespace std;
using namespace cv;

int main()
{
	// The purpose of the mvv_validate .exe is to:
	// -validate .mvv/.bin files generated by mvv_demo
	// -render these files using openCV

	cout << "Beginning mvv validate" << endl;
	

	string file_path = "..\\..\\data_store\\images\\david_1.jpg";
	//string file_path_2 = "C:\Users\Danny\Documents\GitHub\mvv\data_store\images\david_1.jpg";
	Mat tester = imread(file_path, IMREAD_GRAYSCALE);


	cin.get();

    return 0;
}

