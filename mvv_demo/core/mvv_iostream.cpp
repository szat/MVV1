#include "mvv_iostream.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <chrono>
#include <vector>

using namespace std;
using namespace cv;

// what IO functions should we have:

double** convert_vector_params(vector<Mat> forward_params, vector<Mat> reverse_params) {
	int num_triangles = forward_params.size();
	double** params = new double*[num_triangles];
	for (int i = 0; i < num_triangles; i++) {
		double* param_line = new double[12];
		param_line[0] = forward_params[i].at<double>(0, 0);
		param_line[1] = forward_params[i].at<double>(0, 1);
		param_line[2] = forward_params[i].at<double>(0, 2);
		param_line[3] = forward_params[i].at<double>(1, 0);
		param_line[4] = forward_params[i].at<double>(1, 1);
		param_line[5] = forward_params[i].at<double>(1, 2);
		param_line[6] = reverse_params[i].at<double>(0, 0);
		param_line[7] = reverse_params[i].at<double>(0, 1);
		param_line[8] = reverse_params[i].at<double>(0, 2);
		param_line[9] = reverse_params[i].at<double>(1, 0);
		param_line[10] = reverse_params[i].at<double>(1, 1);
		param_line[11] = reverse_params[i].at<double>(1, 2);
		cout << forward_params[i].at<double>(0, 0) << ",";
		cout << reverse_params[i].at<double>(0, 0) << ",";
		params[i] = param_line;
	}
	return params;
}

void save_csv(string folder_name, string file_name, double** affine_params, int num_triangles) {
	ofstream csv_file;
	string full_path = "../data_store/" + folder_name + "/" + file_name;
	csv_file.open(full_path);
	csv_file << num_triangles << "\n";
	for (int i = 0; i < num_triangles; i++) {
		// affine parameters (12 values, 6 forward, 6 reverse)
		int num_affine_params = 12;
		for (int j = 0; j < num_affine_params; j++) {
			csv_file << affine_params[i][j] << ",";
		}
		csv_file << "\n";
	}
	csv_file.close();
}

float** read_csv(string folder_name, string file_name) {
	// this is very inefficient

	string full_path = "../data_store/" + folder_name + "/" + file_name;

	ifstream csv_file;
	csv_file.open(full_path);//open the input file

	stringstream str_stream;
	str_stream << csv_file.rdbuf();//read the file
	//string str = strStream.str();//str holds the content of the file


	int triangles;
	string line;
	getline(str_stream, line);
	triangles = stoi(line);
	float** params = new float*[triangles];

	for (int i = 0; i < triangles; i++) {
		float* param_list = new float[12];
		getline(str_stream, line);
		std::stringstream  lineStream(line);
		std::string        cell;
		int j = 0;
		while (std::getline(lineStream, cell, ','))
		{
			const char *char_cell = cell.c_str();
			float f = atof(char_cell);
			param_list[j] = f;
			j++;
		}
		params[i] = param_list;
	}


	return params;
}

void save_grayscale_t(string folder_name, string file_name, short** frame_triangle_grid, int width, int height) {
	ofstream csv_file;
	string full_path = "../data_store/" + folder_name + "/" + file_name;
	csv_file.open(full_path);
	csv_file << width << "\n";
	csv_file << height << "\n";
	for (int i = 0; i < height; i++) {
		// affine parameters (12 values, 6 forward, 6 reverse)
		for (int j = 0; j < width; j++) {
			csv_file << frame_triangle_grid[i][j] << ",";
		}
		csv_file << "\n";
	}
	csv_file.close();
	//grid[y][x] = i;
}

int** read_grayscale_t(string folder_name, string file_name) {
	// this is very inefficient

	string full_path = "../data_store/" + folder_name + "/" + file_name;

	ifstream csv_file;
	csv_file.open(full_path);//open the input file

	stringstream str_stream;
	str_stream << csv_file.rdbuf();//read the file
								   //string str = strStream.str();//str holds the content of the file
	string str = str_stream.str();
	istringstream iss(str);
	string token;

	getline(iss, token, '\n');
	int width = atoi(token.c_str());
	getline(iss, token, '\n');
	int height = atoi(token.c_str());

	int ** frame = new int*[height];
	for (int i = 0; i < height; i++) {
		int * frame_line = new int[width];
		getline(iss, token, '\n');
		istringstream istringstream_sub(token);
		string sub_token;
		for (int j = 0; j < width; j++) {
			getline(istringstream_sub, sub_token, ',');
			int cell = atoi(sub_token.c_str());
			frame_line[j] = cell;
		}
	}

	return frame;
}

int io_test() {
	// header is 64 bytes
	int num_triangles = 300;
	int num_affine_params = 12;

	float** floats = new float*[num_triangles];
	for (int i = 0; i < num_triangles; i++) {
		floats[i] = new float[num_affine_params];
		for (int j = 0; j < num_affine_params; j++) {
			floats[i][j] = (float)i * 0.3425 * (float)j;
		}
	}
	//save_csv("csv", "triangles.csv", floats, num_triangles);
	float** result = read_csv("csv", "triangles.csv");
	return 0;
}

