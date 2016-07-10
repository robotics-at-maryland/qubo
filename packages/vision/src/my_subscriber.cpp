#include "vision_core.h"

vision_node::vision_node(int argc, char **argv, int rate) {
	image_transport::ImageTransport it(nh);
	camera_sub = it.subscribe("in_image_base_topic", 1, &vision_node::imageCallback, this);
}

vision_node::~vision_node() {} 

void vision_node::imageCallback(const sensor_msgs::ImageConstPtr& msg) {


	// this is creating image pointer
	cv_bridge::CvImageConstPtr cv_ptr;

	// creating a pointer to the image
	try {
		cv_ptr = cv_bridge::toCvCopy(msg, 
				sensor_msgs::image_encodings::BGR8);
	} catch (cv_bridge::Exception& e) {
		ROS_ERROR("cv_bridge exception: %s", e.what());
		return;
	}
	raw_image = cv_ptr->image;
	if (raw_image.empty()) {
		ROS_ERROR("No new image");
	}

	// getting it into proper format

	// canny edge detection 
	cv::cvtColor(raw_image, edge_image, CV_BGR2GRAY);
	cv::blur(edge_image, edge_image, cv::Size(blur_number, blur_number));
	cv::Canny(edge_image, edge_image, canny_low, canny_high, 3);
	dst = cv::Scalar::all(0);
	raw_image.copyTo(dst, edge_image);

	// HSV thresholding
	cv::cvtColor(raw_image, thresh_image, CV_BGR2HSV);
	cv::inRange(thresh_image, cv::Scalar(hue_min, 0, 0), cv::Scalar(hue_max, 255, 255), thresh_image);
	thresh_color = cv::Scalar::all(0);
	raw_image.copyTo(thresh_color, thresh_image);

	cv::imshow("edge image", dst);
	cv::imshow("raw image", raw_image);
	cv::imshow("HSV threshold", thresh_color);

	cv::waitKey(1);
}

void vision_node::update(){
	ros::spinOnce();
	ros::Duration(0.1).sleep();
}


