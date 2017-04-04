#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

int main()
{
    Mat image;// new blank image
    image = cv::imread("test.png", 0);// read the file
    string ty = type2str(image.type());

    namedWindow( "Display window", CV_WINDOW_AUTOSIZE );// create a window for display.
    imshow( "Display window", image );// show our image inside it.
    waitKey(0);// wait for a keystroke in the window
    return 0;
}
