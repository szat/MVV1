// io_demo.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <iostream>
#include <fstream>
using namespace std;

int main() {
	char arr[][3] = { { '1','2','3' },{ '4','5','6' },{ '7','8','9' } };
	char arr2[3][3];
	ofstream out("io3.txt");
	out.write(arr[0], 9);
	out.close();
	ifstream in("io3.txt");
	in.read(arr2[0], 9);
	for (int i = 0; i < 3; ++i)
		for (int j = 0; j < 3; ++j)
			cout << arr2[i][j];
	cout << endl;
}