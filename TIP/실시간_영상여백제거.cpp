#include <opencv2\opencv.hpp>
#include <opencv2\highgui.hpp>

using namespace cv;
using namespace std;

bool cut_margin_realtime(void* arr, int length, int threshold) {
	Mat vec(1, length, CV_16U, arr);

	double min;

	minMaxIdx(vec, &min, NULL);

	if (min > threshold)
		return true;
	else
		return false;
}