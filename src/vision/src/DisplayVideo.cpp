#include "opencv2/opencv.hpp"

using namespace cv;

int main(int, char**)
{
	VideoCapture cap("/dev/fw0"); // open the default camera
	if(!cap.isOpened())  // check if we succeeded
		return -1;

	for(;;)
	{
		Mat frame;
		cap >> frame;
		imshow("Feed", frame);
		waitKey(1); //shows each image for 1 ms
	}
	// the camera will be deinitialized automatically in VideoCapture destructor
	return 0;
}
