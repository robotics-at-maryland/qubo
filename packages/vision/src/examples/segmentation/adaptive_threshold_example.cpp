#include "ros/ros.h"
#include "ros/package.h"
#include "cv.h"
#include "highgui.h"
#include <iostream>
#include <cmath>
#include "opencv2/opencv.hpp"


using namespace cv;

int main(int argc, char **argv) {
	ros::init(argc, argv, "adaptive_threshold_example");
	ros::NodeHandle n;

	// open up the video
	//VideoCapture stream("/home/michael/Documents/ram/data/misc/20100715095315.avi");
	// VideoCapture stream(0);
	VideoCapture stream(ros::package::getPath("vision") + "/data/buoy_backing_up.avi");

	if (!stream.isOpened()) {
		ROS_ERROR("Cannot connect a camera or file! Shuting down...");
		return -1;
	}

	// various stages of images used in processing
	Mat raw_image, thresh_image, thresh_color;

	// process frames at 15hz
	ros::Rate loop_rate(15);

	// variables
	int frame_count = 0,  // a frame count is needed to loop the video
	    blur_number = 6,  // the rest are parameters to the algorithms...
	    kernel_size     = 1;


	namedWindow("raw image", 2);
	namedWindow("HSV threshold", 2);
	createTrackbar("Hue min", "HSV threshold", &kernel_size, 100);

	// main loop
	while (ros::ok()) {
		// get the next frame
		stream >> raw_image;

		// loop the video if we just got the last frame
		if (stream.get(CV_CAP_PROP_FRAME_COUNT) == ++frame_count) {
			frame_count = 0;
			stream.set(CV_CAP_PROP_POS_AVI_RATIO , 0);
		}

		// make sure we actual got an image
		if (raw_image.empty()) {
			ROS_ERROR("Could not read image from VideoCapture!");
			return -1;
		}

		// HSV thresholding
		medianBlur(raw_image, raw_image, pow(2, kernel_size) + 1);
		cvtColor(raw_image, thresh_image, CV_BGR2GRAY);


		adaptiveThreshold(thresh_image, thresh_image, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 11, 2);
		// th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
		//            cv2.THRESH_BINARY,11,2)



		// display images is seperate windows
		imshow("raw image", raw_image);
		imshow("HSV threshold", thresh_image);

		// needed for the opencv GUI
		waitKey(1);

		// make sure we run at 15hz
		loop_rate.sleep();
	}

	destroyWindow("Image");
	return 0;
}
