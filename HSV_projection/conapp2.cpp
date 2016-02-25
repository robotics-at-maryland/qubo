

#include <iostream>
#include <string>
#include <sstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


//Histogram back projction algorithm with Color tracking
using namespace cv;
using namespace std;
int main(int argc, char**argv) {
	VideoCapture cap(0); // opens the video camera. It starts with 0

	//this tests if there is no camera
	if (!cap.isOpened())
	{
		cout << "Cannot open the video cam" << endl;
		return -1;
	}
	double dWidth = cap.get(CV_CAP_PROP_FRAME_WIDTH); //gets the width of frames from the video
	double dHeight = cap.get(CV_CAP_PROP_FRAME_HEIGHT);//get the height of frames from the video

	cout << "Frame size:" << dWidth << "x" << dHeight << endl;
	namedWindow("window", CV_WINDOW_AUTOSIZE);//CREATE A WINDOW CALLED WINDOW
	
	int iLowH = 0;
	int iHighH = 179;

	int ilowS = 0;
	int iHighS = 255;

	int iLowV = 0;
	int iHighV = 255;

	//create trackbars in the window
	cvCreateTrackbar("LowH", "window", &iLowH, 179);//hue(0-179)
	cvCreateTrackbar("HighH", "window", &iHighH, 179);
	cvCreateTrackbar("LowS", "window", &ilowS, 0);//saturation(0-255)
	cvCreateTrackbar("iHighS", "window", &iHighS, 255);
	cvCreateTrackbar("lowV", "window", &iLowV, 0);//value(0-255)
	cvCreateTrackbar("HighV", "window", &iHighV, 255);

	int iLastX = -1;
	int iLastY = -1;

	//Capture a temporary image from the camera
	Mat imgTmp;
	cap.read(imgTmp);

	//create a black image with the size as the camera output
	Mat imgLines = Mat::zeros(imgTmp.size(), CV_8UC3);;

	while (1)
	{
		Mat frame;
		bool bSuccess = cap.read(frame); // read new frame from video

		if (!bSuccess)
		{
			cout << "Cannot read a frame video stream" << endl;
			break;
		}
		imshow("window", frame);

		Mat imgHSV;
		cvtColor(frame, imgHSV, COLOR_BGR2HSV);//CONVERT THE CAPTURE FRAME FROM BGR TO HSV
		
		Mat imgThresholded;
		inRange(imgHSV, Scalar(iLowH, ilowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded);//threshold the image
		//morphological opening (remove small object from the foreground
		erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
		dilate(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));

		//morphological closing(fill small holes in the forground)
		dilate(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
		erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));

		//Calcualte the moments of the thresholded image
		Moments oMoments = moments(imgThresholded);

		double dM01 = oMoments.m01;
		double dM10 = oMoments.m10;
		double dArea = oMoments.m00;

		//if the area <= 10000, i consider that there are no object in the image and it's becuase of the noise, the area is not zero

		if (dArea > 10000)
		{
			//calculate the position of the ball
			int posX = dM10 / dArea;
			int posY = dM01 / dArea;

			if (iLastX >= 0 && iLastY >= 0 && posX >= 0 && posY >= 0) {
				//Draw a red line from the previous point to the current point
				line(imgLines, Point(posX, posY), Point(iLastX, iLastY), Scalar(0, 0, 255), 2);
			}
			iLastX = posX;
			iLastY = posY;

		}

		frame = frame + imgLines;
		imshow("Thresholded image", imgThresholded);
		imshow("frame", frame);

		if (waitKey(30) == 27)
		{
			cout << "esc key is pressed by user" << endl;
			break;
		}
	}
	
	return 0;
}
