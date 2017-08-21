#pragma once
#include <string>
#include <iostream>
#include "INIReader.h"
#include "windows.h"

#define SETTINGS_PATH "../../data_store/settings.ini"

int get_video_width() {
	INIReader reader(SETTINGS_PATH);
	if (reader.ParseError() < 0) {
		std::cout << "ERROR CODE 001: Can't load 'settings.ini'\n";
		return 0;
	}
	return reader.GetInteger("user", "video_width", 0);
}

int get_video_height() {
	INIReader reader(SETTINGS_PATH);
	if (reader.ParseError() < 0) {
		std::cout << "ERROR CODE 001: Can't load 'settings.ini'\n";
		return 0;
	}
	return reader.GetInteger("user", "video_height", 0);
}