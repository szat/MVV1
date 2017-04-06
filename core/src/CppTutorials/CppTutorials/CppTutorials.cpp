// CppTutorials.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <iostream>
#include <vector>
#include <stack>
#include <ctime>


using namespace std;

std::stack<clock_t> tictoc_stack;

void tic() {
	tictoc_stack.push(clock());
}

void toc() {
	cout << "Time elapsed: "
		<< ((double)(clock() - tictoc_stack.top())) / CLOCKS_PER_SEC
		<< endl;
	tictoc_stack.pop();
}

int main()
{

	

	vector<int> vec = { 1 };

	printf("Creating vector");
	tic();
	for (int i = 0; i < 1000; i++) {
		vec.push_back(1);
	}
	toc();

	for (int j = 0; j < 10; j++) {
		printf("Simon's method");
		tic();
		for (int i = 0; i < vec.size(); i++) {
			if (i % 2 == 0) {
				vec.push_back(0);
			}
		}
		toc();

		printf("Danny's method");
		tic();
		int vecSize = vec.size();
		for (int i = 0; i < vecSize; i++) {
			int test = vec[i];
			if (i % 2 == 0) {
				vec.push_back(0);
			}
		}
		toc();
	}
	
	cin.get();

    return 0;
}

