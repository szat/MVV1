#include "flash_detection.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/objdetect.hpp"
#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgcodecs.hpp>
#include <vector>
#include <string>
#include <iostream>

using namespace std;
using namespace cv;


int calculate_intensity_maxima(vector<float> intensity) {
	int len = intensity.size();
	float max_diff = 0;
	int max_diff_index = 0;
	for (int i = 0; i < len - 1; i++) {
		float new_diff = intensity[i + 1] - intensity[i];
		if (new_diff > max_diff) {
			max_diff = new_diff;
			max_diff_index = i + 1;
		}
	}

	return max_diff_index;
}

void get_frame_intensity(Mat &frame, vector<float> &intensity_values, int width, int height) {
	Mat intensity_frame;
	cvtColor(frame, intensity_frame, CV_BGR2Lab);
	Mat channels[3];
	split(intensity_frame, channels);
	Scalar intensity_average = mean(channels[0]);
	cout << intensity_average << endl;
	float intensity = intensity_average[0, 0];
	intensity_values.push_back(intensity);
}

int get_flash_maxima(string video_path) {

	VideoCapture capture(video_path); // open the default camera
	if (!capture.isOpened())  // check if we succeeded
		cout << "Error opening video" << endl;
		return 0;

	int width = capture.get(CV_CAP_PROP_FRAME_WIDTH);
	int height = capture.get(CV_CAP_PROP_FRAME_HEIGHT);

	namedWindow("frame", 1);
	vector<float> intensity = vector<float>();
	Mat frame;
	for (;;)
	{
		if (!capture.read(frame)) {
			cout << "Error reading frame";
			return 0;
		}
		get_frame_intensity(frame, intensity, width, height);
		//if (waitKey(30) >= 0) break;
	}

	int flash_frame = calculate_intensity_maxima(intensity);
	return flash_frame;
}


pair<int,int> get_flash_timing(string video_directory, string video_path_1, string video_path_2) {
	pair<int, int> maxima_timing = pair<int, int>(0, 0);

	string full_path_1 = video_directory + video_path_1;
	string full_path_2 = video_directory + video_path_2;

	int flash_maxima_1 = get_flash_maxima(full_path_1);
	int flash_maxima_2 = get_flash_maxima(full_path_2);

	maxima_timing = pair<int, int>(flash_maxima_1, flash_maxima_2);
	return maxima_timing;
}