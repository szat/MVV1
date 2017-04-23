#include "knn_test.h"

using namespace std;
using namespace cv;

void knn_test(vector<Point2f> points2D) {

	flann::KDTreeIndexParams indexParams;
	flann::Index kdtree(Mat(points2D).reshape(1), indexParams);
	vector<float> query = { points2D.at(0).x, points2D.at(0).y };

	vector<int> indices;
	vector<float> dists;
	float radius = 100;
	int nb_points = 100;
	cout << "We are in knn_test!" << endl;
	cout << "kdtree.radiusSearch((" << points2D.at(0).x << ", " << points2D.at(0).y << "), indices, dists, radius = " << radius << ", nb_points = " << nb_points << ")" << endl;
	kdtree.radiusSearch(query, indices, dists, radius, nb_points); //big radius and small number of neighboors should give all k the neighboors
	cout << "The query point is (" << points2D.at(0).x << "," << points2D.at(0).y << "), that's how we roll" << endl;
	vector<Point2f> nbh; 
	for (int i = 0; i < indices.size(); i++) {
		nbh.push_back(points2D.at(indices.at(i)));
		float dist = (points2D.at(indices.at(i)).x - points2D.at(0).x)*(points2D.at(indices.at(i)).x - points2D.at(0).x) +  (points2D.at(indices.at(i)).y - points2D.at(0).y)*(points2D.at(indices.at(i)).y - points2D.at(0).y);
		cout << "The " << i << "th points is (" << points2D.at(indices.at(i)).x << "," << points2D.at(indices.at(i)).y << "), dist^2 = " << sqrt(dist) << endl;
	}

	vector<int> indices2;
	vector<float> dists2;
	cout << endl;
	cout << "kdtree.knnSearch((" << points2D.at(0).x << ", " << points2D.at(0).y << "), indices2, dists2, int = " << nb_points << ")" << endl;
	kdtree.knnSearch(query, indices2, dists2, 20);
	cout << "The query point is (" << points2D.at(0).x << "," << points2D.at(0).y << "), that's how we roll" << endl;
	vector<Point2f> nbh2;
	for (int i = 0; i < indices2.size(); i++) {
		nbh2.push_back(points2D.at(indices2.at(i)));
		float dist = (points2D.at(indices2.at(i)).x - points2D.at(0).x)*(points2D.at(indices2.at(i)).x - points2D.at(0).x) + (points2D.at(indices2.at(i)).y - points2D.at(0).y)*(points2D.at(indices2.at(i)).y - points2D.at(0).y);
		cout << "The " << i << "th points is (" << points2D.at(indices2.at(i)).x << "," << points2D.at(indices2.at(i)).y << "), dist^2 = " << sqrt(dist) << endl;
	}



}