#include "vision_core.h"

vision_node::vision_node(int argc, char **argv, int rate) {
  sub = it.subscribe("/camera/msg", 1, imageCallback);
}

void imageCallback(const sensor_msgs::ImageConstPtr& msg) {
    cv_bridge::CvImagePtr cv_ptr;
    try {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    } catch (cv_bridge::Exception& e) {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }

    // make sure we actual got an image
    if (raw_image.empty()) {
      ROS_ERROR("Could not read image from VideoCapture!");
      return -1;
    }

    // Canny edge detection
    cvtColor(raw_image, edge_image, CV_BGR2GRAY);
    blur(edge_image, edge_image, Size(blur_number, blur_number));
    Canny(edge_image, edge_image, canny_low, canny_high, 3);
    dst = Scalar::all(0);
    raw_image.copyTo(dst, edge_image);

    // HSV thresholding
    cvtColor(raw_image, thresh_image, CV_BGR2HSV);
    inRange(thresh_image, Scalar(hue_min, 0, 0), Scalar(hue_max, 255, 255), thresh_image);
    thresh_color = Scalar::all(0);
    raw_image.copyTo(thresh_color, thresh_image);

    // orb features
    cvtColor(raw_image, gray_image, CV_BGR2GRAY);
    std::vector<KeyPoint> orb_key_points;
    orb(gray_image, gray_image, orb_key_points, noArray());
    drawKeypoints(raw_image, orb_key_points, raw_image, Scalar(0, 0, 255));

    // display images is seperate windows
    imshow("edge image", dst);
    imshow("raw image", raw_image);
    imshow("HSV threshold", thresh_color);

    // needed for the opencv GUI
    waitKey(1);
}

void update(){
  ros::spinOnce();
  ros::Duration(0.1).sleep();
}
