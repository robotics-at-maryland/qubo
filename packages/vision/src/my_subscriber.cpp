#include "vision_core.h"

void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
  try
  {

    raw_image = msg;
    // Canny edge detection
    cv::cvtColor(raw_image, edge_image, CV_BGR2GRAY);
    cv::blur(edge_image, edge_image, cv::Size(blur_number, blur_number));
    cv::Canny(edge_image, edge_image, canny_low, canny_high, 3);
    dst = cv::Scalar::all(0);
    raw_image.copyTo(dst, edge_image);

    // HSV thresholding
    cv::cvtColor(raw_image, thresh_image, CV_BGR2HSV);
    cv::inRange(thresh_image, cv::Scalar(hue_min, 0, 0), cv::Scalar(hue_max, 255, 255), thresh_image);
    thresh_color = Scalar::all(0);
    raw_image.copyTo(thresh_color, thresh_image);

    // orb features
    cv::cvtColor(raw_image, gray_image, CV_BGR2GRAY);
    cv::orb(gray_image, gray_image, orb_key_points, noArray());
    cv::drawKeypoints(raw_image, orb_key_points, raw_image, Scalar(0, 0, 255));

    // display images is seperate windows
    cv::imshow("edge image", dst);
    cv::imshow("raw image", raw_image);
    cv::imshow("HSV threshold", thresh_color);
    cv::imshow("view", cv_bridge::toCvShare(msg, "bgr8")->image);
    cv::waitKey(30);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
  }
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "image_listener");
  ros::NodeHandle nh;
  cv::namedWindow("view");
  cv::namedWindow("edge image", 2);
  cv::createTrackbar("canny low", "edge image", &canny_low, 200);
  cv::createTrackbar("canny high", "edge image", &canny_high, 200);
  cv::createTrackbar("blur size", "edge image", &blur_number, 50);

  cv::namedWindow("raw image", 2);
  cv::namedWindow("HSV threshold", 2);
  cv::createTrackbar("Hue min", "HSV threshold", &hue_min, 255);
  cv::createTrackbar("Hue max", "HSV threshold", &hue_max, 255);

  cv::startWindowThread();
  image_transport::ImageTransport it(nh);
  image_transport::Subscriber sub = it.subscribe("camera/image", 1, imageCallback);
  ros::spin();
  cv::destroyWindow("view");
}
