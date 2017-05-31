#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
//C
#include <stdio.h>
//C++
#include <iostream>
#include <sstream>

using namespace cv;
using namespace std;

#include "background_substraction.h"

int test_bs() {
	Mat frame; //current frame
	Mat fgMaskMOG2; //fg mask fg mask generated by MOG2 method
	Ptr<BackgroundSubtractor> pMOG2; //MOG2 Background subtractor
	int keyboard; //input from keyboard
	cout << "We are in test_bs()" << endl;

	namedWindow("Frame");
	namedWindow("FG Mask");
	pMOG2 = createBackgroundSubtractorMOG2(); //MOG2 approach

	//Open Video File
	string first_frame_path = "../../data_store/background_substraction/1.png"; 
	//read the first file of the sequenc
	frame = imread(first_frame_path);
	if (frame.empty()) {
		//error in opening the first image
		cerr << "Unable to open first image frame: " << first_frame_path << endl;
		exit(EXIT_FAILURE);
	}
	//current image filename
	string fn(first_frame_path);
	//read input data. ESC or 'q' for quitting
	while ((char)keyboard != 'q' && (char)keyboard != 27) {
		//update the background model
		pMOG2->apply(frame, fgMaskMOG2);
		//get the frame number and write it on the current frame
		size_t index = fn.find_last_of("/");
		if (index == string::npos) {
			index = fn.find_last_of("\\");
		}
		size_t index2 = fn.find_last_of(".");
		string prefix = fn.substr(0, index + 1);
		string suffix = fn.substr(index2);
		string frameNumberString = fn.substr(index + 1, index2 - index - 1);
		istringstream iss(frameNumberString);
		int frameNumber = 0;
		iss >> frameNumber;
		rectangle(frame, cv::Point(10, 2), cv::Point(100, 20),
			cv::Scalar(255, 255, 255), -1);
		putText(frame, frameNumberString.c_str(), cv::Point(15, 15),
			FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
		//show the current frame and the fg masks
		imshow("Frame", frame);
		imshow("FG Mask MOG 2", fgMaskMOG2);
		//get the input from the keyboard
		keyboard = waitKey(30);
		//search for the next image in the sequence
		ostringstream oss;
		oss << (frameNumber + 1);
		string nextFrameNumberString = oss.str();
		string next_frame_path = prefix + nextFrameNumberString + suffix;
		//read the next frame
		frame = imread(next_frame_path);
		if (frame.empty()) {
			//error in opening the next image in the sequence
			cerr << "Unable to open image frame: " << next_frame_path << endl;
			exit(EXIT_FAILURE);
		}
		//update the path of the current frame
		fn.assign(next_frame_path);
	}

	return 0;
}
