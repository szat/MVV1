#pragma once
#include <string>
#include <iostream>
#include "INIReader.h"
#include "windows.h"

#define SETTINGS_PATH "../../data_store/settings.ini"

INIReader initialize_reader() {
	INIReader reader(SETTINGS_PATH);
	if (reader.ParseError() < 0) {
		std::cout << "ERROR CODE 001: Can't load 'settings.ini'\n";
		std::cout << "Press enter to exit program.\n";
		std::cin.get();
		exit(EXIT_FAILURE);
	}
	return reader;
}

int get_video_width() {
	INIReader reader = initialize_reader();
	return reader.GetInteger("user", "video_width", 0);
}

int get_video_height() {
	INIReader reader = initialize_reader();
	return reader.GetInteger("user", "video_height", 0);
}

string get_video_path_1() {
	INIReader reader = initialize_reader();
	return reader.Get("user", "video_path_1", "UNKNOWN");
}

string get_video_path_2() {
	INIReader reader = initialize_reader();
	return reader.Get("user", "video_path_2", "UNKNOWN");
}

int get_start_offset() {
	INIReader reader = initialize_reader();
	return reader.GetInteger("user", "start_offset", 0);
}

int get_framerate() {
	INIReader reader = initialize_reader();
	return reader.GetInteger("user", "framerate", 0);
}

float get_delay() {
	INIReader reader = initialize_reader();
	double double_delay = reader.GetReal("user", "pi", -1);
	float delay = (float)double_delay;
	return delay;
}

