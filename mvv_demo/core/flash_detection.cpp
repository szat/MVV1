#include "flash_detection.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <string>

using namespace std;
using namespace cv;



int get_flash_maxima(string video_path) {


	return 64;
}


pair<int,int> test_flash(string video_directory, string video_path_1, string video_path_2) {

	string full_path_1 = video_directory + video_path_1;
	string full_path_2 = video_directory + video_path_2;



	int flash_maxima_1 = get_flash_maxima(full_path_1);
	int flash_maxima_2 = get_flash_maxima(full_path_2);

	pair<int, int> maxima_timing = pair<int, int>(flash_maxima_1, flash_maxima_2);
	return maxima_timing;
}