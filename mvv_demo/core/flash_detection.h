#pragma once

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <string>

std::pair<int, int> test_flash(std::string video_directory, std::string video_path_1, std::string video_path_2);