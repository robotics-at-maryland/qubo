//#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>
//#include <opencv2/opencv.hpp>

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
//#include <sstream> // for converting the command line parameter to integer


//int main(int argc, char** argv)
//{
//	// Check if video source has been passed as a parameter
//	if(argv[1] == NULL) return 1;
//
//	ros::init(argc, argv, "image_publisher");
//	ros::NodeHandle nh;
//	image_transport::ImageTransport it(nh);
//	image_transport::Publisher pub = it.advertise("camera/image", 1);
//
//	// Convert the passed as command line parameter index for the video device to an integer
//	std::istringstream video_sourceCmd(argv[1]);
//	int video_source;
//	// Check if it is indeed a number
//	if(!(video_sourceCmd >> video_source)) return 1;
//
//	cv::VideoCapture cap(video_source);
//	// Check if video device can be opened with the given index
//	if(!cap.isOpened()) return 1;
//	cv::Mat frame;
//	sensor_msgs::ImagePtr msg;
//
//	ros::Rate loop_rate(5);
//	while (nh.ok()) {
//		cap >> frame;
//		// Check if grabbed frame is actually full with some content
//		if(!frame.empty()) {
//			msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", frame).toImageMsg();
//			pub.publish(msg);
//			cv::waitKey(1);
//		}
//
//		ros::spinOnce();
//		loop_rate.sleep();
//	}
//}

using namespace cv;

int main(int argc, char** argv)
{ 

	ros::init(argc, argv, "image_publisher");
	ros::NodeHandle nh;
	image_transport::ImageTransport it(nh);
	image_transport::Publisher pub = it.advertise("camera/image", 1);
	sensor_msgs::ImagePtr msg;

	// creates a video capture from default camera
	VideoCapture cap (0);

	if(!cap.isOpened()){
		return -1;
	}

	// creates a window
	Mat edges;
	namedWindow("edges",1);

	//the rate at which we loop
	ros::Rate loop_rate(5);

	// while the node is hanging on
	while(nh.ok())
	{

		Mat frame;
		cap >> frame; // get a new frame from camera


		cvtColor(frame, edges, CV_BGR2GRAY);
		GaussianBlur(edges, edges, Size(7,7), 1.5, 1.5);
		Canny(edges, edges, 0, 30, 3);
		imshow("edges", edges);

		// Check if grabbed frame is actually full with some content
		if(!frame.empty()) {
			msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", frame).toImageMsg();
			pub.publish(msg);
			cv::waitKey(1);
		}

		ros::spinOnce();
		loop_rate.sleep();
		if(waitKey(30) >= 0) break;

		// let's see if we can avoid the try catch with using videocapture methods
	}
	// the camera will be deinitialized automatically in VideoCapture destructor
	return 0;	
}

