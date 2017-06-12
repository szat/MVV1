#include "video_preprocessing.h"
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
	/*
	Takes an input of per-frame intensity and returns the index
	at which the greatest step difference occurs. We will have to
	do some testing, but this seems like a decently reliable way
	of figuring out where the flash occurs.
	*/
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
	cout << "Max diff: " << max_diff << endl;
	cout << "Max frame: " << max_diff_index << endl;

	return max_diff_index;
}

void get_frame_intensity(Mat &frame, vector<float> &intensity_values, int width, int height) {
	Mat intensity_frame;
	cvtColor(frame, intensity_frame, CV_BGR2Lab);
	// Here the CIELAB color space is used, with the Luminosity
	// channel being taken as the 'intensity' of the frame.
	Mat channels[3];
	split(intensity_frame, channels);
	Scalar intensity_average = mean(channels[0]);
	float intensity = intensity_average[0, 0];
	intensity_values.push_back(intensity);
}

int get_flash_maxima(string video_path) {
	VideoCapture capture(video_path); // open the default camera
	if (!capture.isOpened()) { // check if we succeeded
		cout << "Error opening video" << endl;
		return 0;
	}

	int width = capture.get(CV_CAP_PROP_FRAME_WIDTH);
	int height = capture.get(CV_CAP_PROP_FRAME_HEIGHT);

	vector<float> intensity = vector<float>();
	Mat frame;
	for (;;)
	{
		if (!capture.read(frame)) {
			cout << "Error reading frame" << endl;
			break;
		}
		get_frame_intensity(frame, intensity, width, height);
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

void save_trimmed_videos(pair<int,int> flash_result, string input_dir, string output_dir, string input_1, string input_2, string output_1, string output_2) {
	cout << "Trimming videos" << endl;
	string input_path_1 = input_dir + input_1;
	string input_path_2 = input_dir + input_2;
	string output_path_1 = output_dir + output_1;
	string output_path_2 = output_dir + output_2;

	VideoCapture in_capture_1(input_path_1);
	Mat img;

	if (!in_capture_1.isOpened()) {  // check if we succeeded
		cout << "Error opening video" << endl;
	}

	//CV_CAP_PROP_FRAME_COUNT


	int width_1 = in_capture_1.get(CV_CAP_PROP_FRAME_WIDTH);
	int height_1 = in_capture_1.get(CV_CAP_PROP_FRAME_HEIGHT);
	int length_1 = in_capture_1.get(CV_CAP_PROP_FRAME_COUNT);

	VideoWriter out_capture_1(output_path_1, CV_FOURCC('D', 'I', 'V', 'X'), 30, Size(width_1, height_1));

	while (true)
	{
		in_capture_1 >> img;
		if (img.empty())
			break;

		out_capture_1.write(img);
	}

}