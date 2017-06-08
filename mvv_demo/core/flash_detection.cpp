#include "flash_detection.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/objdetect.hpp"
#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgcodecs.hpp>
#include <vector>
#include <string>

using namespace std;
using namespace cv;



int get_flash_maxima(string video_path) {


	return 64;
}


pair<int,int> test_flash(string video_directory, string video_path_1, string video_path_2) {
	pair<int, int> maxima_timing = pair<int, int>(0, 0);

	string full_path_1 = video_directory + video_path_1;
	string full_path_2 = video_directory + video_path_2;



	VideoCapture capture(full_path_1); // open the default camera
	if (!capture.isOpened())  // check if we succeeded
		return maxima_timing;

	Mat edges;
	namedWindow("edges", 1);
	Mat frame;
	for (;;)
	{
		if (!capture.read(frame)) {
			break;
		}
		cvtColor(frame, edges, CV_BGR2GRAY);
		GaussianBlur(edges, edges, Size(7, 7), 1.5, 1.5);
		Canny(edges, edges, 0, 30, 3);
		imshow("edges", edges);
		waitKey(15);
//		if (waitKey(30) >= 0) break;
	}



	int flash_maxima_1 = get_flash_maxima(full_path_1);
	int flash_maxima_2 = get_flash_maxima(full_path_2);

	maxima_timing = pair<int, int>(flash_maxima_1, flash_maxima_2);
	return maxima_timing;
}