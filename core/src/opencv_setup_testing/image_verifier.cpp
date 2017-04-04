#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
using namespace cv;
using namespace std;
int main()
{

    Mat image1, image2;
    image1 = imread("lena.bmp", CV_LOAD_IMAGE_COLOR);
    if (!image1.data)
    {
        cout << "Could not open or find the image" << endl;
        return -1;
    }

    image2 = imread("lena.bmp", CV_LOAD_IMAGE_GRAYSCALE);
    if (!image2.data)
    {
        cout << "Could not open or find the image" << endl;
        return -1;
    }

    cout << "opencv test" << endl;
    namedWindow("Color Image", WINDOW_AUTOSIZE);
    imshow("Color Image", WINDOW_AUTOSIZE);
    namedWindow("Gray Scale Image", WINDOW_AUTOSIZE);
    imshow("Gray Scale Image", image2);
    waitKey(0);
    destroyAllWindows();
    return 0;
}
