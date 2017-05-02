#include "optical_flow.h"

using namespace std;
using namespace cv;

void test_discrete_opt_flow(std::string imagePathA, std::string imagePathB) {
	//images must be of the same size
	cout << "Welcome to match_points_2 testing unit!" << endl;
	string address = "..\\data_store\\" + imagePathA;
	string input = "";
	ifstream infile1;
	infile1.open(address.c_str());

	Mat img1 = imread(address, IMREAD_GRAYSCALE);
	Mat img1Display = imread(address);

	address = "..\\data_store\\" + imagePathB;
	input = "";
	ifstream infile2;
	infile2.open(address.c_str());

	Mat img2 = imread(address, IMREAD_GRAYSCALE);
	Mat img2Display = imread(address);

	Ptr<GFTTDetector> detectorGFTT = GFTTDetector::create();
	detectorGFTT->setMaxFeatures(5000);
	detectorGFTT->setQualityLevel(0.00001);
	detectorGFTT->setMinDistance(2);
	detectorGFTT->setBlockSize(3);
	detectorGFTT->setHarrisDetector(true);
	detectorGFTT->setK(0.2);

	//Find features
	vector<KeyPoint> kpts1_gftt;
	detectorGFTT->detect(img1, kpts1_gftt);
	cout << "GFTT found " << kpts1_gftt.size() << " feature points in image A." << endl;

	//Compute descriptors
	Ptr<DescriptorExtractor> extractorORB = ORB::create();
	Mat desc1_new;
	extractorORB->compute(img1, kpts1_gftt, desc1_new);

	//Find features
	vector<KeyPoint> kpts2_gftt;
	detectorGFTT->detect(img2, kpts2_gftt);
	cout << "GFTT found " << kpts2_gftt.size() << " feature points in image B." << endl;

	//Compute descriptors
	Mat desc2_new;
	extractorORB->compute(img2, kpts2_gftt, desc2_new);

	//Optical Flow
	vector<Point2f> initial_points, new_points;
	// fill the initial points vector 
	for (int i = 0; i < kpts1_gftt.size(); i++) {
		initial_points.push_back(kpts1_gftt.at(i).pt);
	}
	std::vector<uchar> status;
	std::vector<float> errors;

	cv::calcOpticalFlowPyrLK(img1, img2, initial_points, new_points, status, errors);
	cout << "Inside test_optical_flow" << endl;

	//Visualize, first put back into KeyPoints
	vector<KeyPoint> kpts2_flow;
	vector<KeyPoint> kpts1_flow;
	for (int i = 0; i < new_points.size(); i++) {
		if (status.at(i) == 1) {
			KeyPoint good;
			good.pt = new_points.at(i);
			kpts2_flow.push_back(good);
			kpts1_flow.push_back(kpts1_gftt.at(i));
		}
	}

	cout << "number of matched features " << kpts1_flow.size() << endl;

	Mat img1to2;
	vector<DMatch> matchesIndexTrivial;

	for (int i = 0; i < kpts2_flow.size(); i++) matchesIndexTrivial.push_back(DMatch(i, i, 0));

	drawMatches(img1Display, kpts1_flow, img2Display, kpts2_flow, matchesIndexTrivial, img1to2);

	//-- Show detected matches
	imshow("Matches", img1to2);

	waitKey(0);
}