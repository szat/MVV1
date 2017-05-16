#pragma once

#include <string>

void save_raster(std::string full_path, short ** raster, int width, int height);

void write_float_array(std::string full_path, float * input, int length);

void test_binary();
