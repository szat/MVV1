// mvv_trial.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

class PointCorrespondence {
	public:
		PointCorrespondence(float AX, float AY, float BX, float BY);
		float imageAX;
		float imageAY;
		float imageBX;
		float imageBY;
};

PointCorrespondence::PointCorrespondence(float AX, float AY, float BX, float BY) {
	imageAX = AX;
	imageAY = AY;
	imageBX = BX;
	imageBY = BY;
}

int main()
{

	cout << "Constructing polygons..." << endl;

	PointCorrespondence testCorr = PointCorrespondence(1,2,3,4);

	vector<PointCorrespondence> testCorrVector = vector<PointCorrespondence>();
	testCorrVector.push_back(testCorr);


	cin.get();

	return 0;
}
