#include "binary_io.h"
#include <iostream>
#include <fstream>
#include <chrono>

using namespace std;

// Note about all of this:
// OS-dependent! This will not work on other architectures (for now).


void write_char_array(string full_path, char * input, int length) {
	ofstream ofile(full_path, ios::binary);
	char * length_array = new char[4];
	for (int k = 0; k < 4; k++) {
		length_array[k] = (length >> k * 8) & 0xFF;
	}
	ofile.write(length_array, 4);
	ofile.write(input, length);
	ofile.close();
}

char * read_char_array(string full_path, int &length) {
	// modifies length, returns char array
	ifstream ifile(full_path, ios::binary);
	char * int_array = new char[4];
	ifile.read(int_array, 4);
	for (int k = 0; k < 4; k++) {
		unsigned char len_char = (unsigned char)int_array[k];
		length += len_char << k * 8;
	}
	char * result = new char[length];
	ifile.read(result, length);
	return result;
}

void write_short_array(string full_path, short * input, int length) {
	char * char_input = new char[length * 2];
	ofstream ofile(full_path, ios::binary);
	char * length_array = new char[4];
	for (int k = 0; k < 4; k++) {
		length_array[k] = (length >> k * 8) & 0xFF;
	}
	for (int i = 0; i < length; i++) {
		char_input[i * 2] = input[i] & 0xFF;
		char_input[i * 2 + 1] = (input[i] >> 8) & 0xFF;
	}
	ofile.write(length_array, 4);
	ofile.write(char_input, length * 2);
}

short * read_short_array(string full_path, int &length) {
	ifstream ifile(full_path, ios::binary);
	char * int_array = new char[4];
	ifile.read(int_array, 4);
	for (int k = 0; k < 4; k++) {
		unsigned char len_char = (unsigned char)int_array[k];
		length += len_char << k * 8;
	}
	char * result = new char[length * 2];
	ifile.read(result, length * 2);
	short * short_result = new short[length];
	for (int i = 0; i < length; i++) {
		short_result[i] = (result[i * 2 + 1] >> 8) ^ result[i * 2];
	}

	return short_result;
}

void write_float_array(string full_path, float * input, int length) {
}


void timing() {
	int len = 1000000;

	short * stuff = new short[len];
	for (int i = 0; i < len; i++) {
		stuff[i] = (short)i;
	}

	char * stuff2 = new char[len];
	for (int i = 0; i < len; i++) {
		stuff2[i] = (char)i;
	}

	auto t1 = std::chrono::high_resolution_clock::now();


	string full_path = "../data_store/foobar.bin";
	write_short_array(full_path, stuff, len);

	auto t2 = std::chrono::high_resolution_clock::now();
	std::cout << "write short took "
		<< std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
		<< " milliseconds\n";

	int length = 0;
	short *result = read_short_array(full_path, length);

	auto t3 = std::chrono::high_resolution_clock::now();
	std::cout << "read short took "
		<< std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count()
		<< " milliseconds\n";

	string full_path2 = "../data_store/foobar2.bin";
	write_char_array(full_path2, stuff2, len);

	auto t4 = std::chrono::high_resolution_clock::now();
	std::cout << "write char took "
		<< std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count()
		<< " milliseconds\n";

	int length2 = 0;
	read_char_array(full_path2, length2);

	auto t5 = std::chrono::high_resolution_clock::now();
	std::cout << "read char took "
		<< std::chrono::duration_cast<std::chrono::milliseconds>(t5 - t4).count()
		<< " milliseconds\n";
}

void test_binary() {
	cout << "Test binary";

	//b[0] = si & 0xff;
	//b[1] = (si >> 8) & 0xff;

	int len = 20000;

	short * stuff = new short[len];
	for (int i = 0; i < len; i++) {
		stuff[i] = (short)i;
	}

	string full_path = "../data_store/foobar.bin";
	write_short_array(full_path, stuff, len);

	int length = 0;
	short *result = read_short_array(full_path, length);

	short result25 = result[25];
	short result1000 = result[1000];
	cout << "Testing";
	// test time for 1 million char, 1 million short
};