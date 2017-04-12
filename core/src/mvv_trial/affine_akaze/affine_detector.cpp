

#include <iostream>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <math.h>

#define M_PI 3.14159265358979323846

using namespace std;
using namespace cv;

void affineSkew(double tilt, double phi, Mat& img, Mat& mask, Mat& Ai)
{
	int h = img.rows;
	int w = img.cols;

	mask = Mat(h, w, CV_8UC1, Scalar(255));

	Mat A = Mat::eye(2, 3, CV_32F);

	if (phi != 0.0)
	{
		phi *= M_PI / 180.;
		double s = sin(phi);
		double c = cos(phi);

		A = (Mat_<float>(2, 2) << c, -s, s, c);

		Mat corners = (Mat_<float>(4, 2) << 0, 0, w, 0, w, h, 0, h);
		Mat tcorners = corners*A.t();
		Mat tcorners_x, tcorners_y;
		tcorners.col(0).copyTo(tcorners_x);
		tcorners.col(1).copyTo(tcorners_y);
		std::vector<Mat> channels;
		channels.push_back(tcorners_x);
		channels.push_back(tcorners_y);
		merge(channels, tcorners);

		Rect rect = boundingRect(tcorners);
		A = (Mat_<float>(2, 3) << c, -s, -rect.x, s, c, -rect.y);

		warpAffine(img, img, A, Size(rect.width, rect.height), INTER_LINEAR, BORDER_REPLICATE);
	}
	if (tilt != 1.0)
	{
		double s = 0.8*sqrt(tilt*tilt - 1);
		GaussianBlur(img, img, Size(0, 0), s, 0.01);
		resize(img, img, Size(0, 0), 1.0 / tilt, 1.0, INTER_NEAREST);
		A.row(0) = A.row(0) / tilt;
	}
	if (tilt != 1.0 || phi != 0.0)
	{
		h = img.rows;
		w = img.cols;
		warpAffine(mask, mask, A, Size(w, h), INTER_NEAREST);
	}
	invertAffineTransform(A, Ai);
}

int main() {
	return 0;
}
