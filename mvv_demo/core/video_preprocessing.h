#pragma once

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <string>

std::pair<int, int> get_flash_timing(std::string video_directory, std::string video_path_1, std::string video_path_2, int stop_frame);

int synchronize_videos(std::string video_directory, std::string video_path_1, std::string video_path_2, int stop_frame);

void save_trimmed_videos(std::pair<int, int> flash_result, std::string input_dir, std::string output_dir, std::string input_1, std::string input_2, std::string output_1, std::string output_2);