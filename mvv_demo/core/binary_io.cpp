#include "binary_io.h"
#include <iostream>
#include <fstream>

using namespace std;


void write_char_array(string full_path, char * input, int length) {
	ofstream ofile(full_path, ios::binary);

	ofile.write(input, length);
	ofile.write((char*)length, 4);
	ofile.close();
}

char * read_char_array(string full_path, int &length) {
	// modifies length, returns char array
	char * result = new char[200];
	ifstream ifile(full_path, ios::binary);
	ifile.read(result, 200);
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

void write_float_array(string full_path, float * input, int length) {
}


void test_binary() {
	cout << "Test binary";

	//b[0] = si & 0xff;
	//b[1] = (si >> 8) & 0xff;

	char * stuff = new char[200];
	for (short i = 0; i < 200; i++) {
		stuff[i] = (char)i;
	}



	string full_path = "../data_store/foobar.bin";
	write_char_array(full_path, stuff, 200);


	// test time for 1 million char, 1 million short
};