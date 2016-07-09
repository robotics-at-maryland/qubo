#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

using namespace cv;

int main(int argc, char** argv)
{
	VideoCapture cap ("dev/fw0");

	if(!cap.isOpened()){
		return -1;
	}

	Mat edges;
	namedWindow("edges",1);
	for(;;)
	{
		try{
			Mat frame;
			cap >> frame; // get a new frame from camera


			cvtColor(frame, edges, CV_BGR2GRAY);
			GaussianBlur(edges, edges, Size(7,7), 1.5, 1.5);
			Canny(edges, edges, 0, 30, 3);
			imshow("edges", edges);
			if(waitKey(30) >= 0) break;

			// let's see if we can avoid the try catch with using videocapture methods
		}catch (std::exception const &exc)
		{
			std::cerr << "PEACE OUT GIRL SCOUT " << exc.what() << "\n";
			return -1;
		}
	}
	// the camera will be deinitialized automatically in VideoCapture destructor
	return 0;	
}
