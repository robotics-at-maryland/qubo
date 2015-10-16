#include "ros/ros.h"
#include "ros/package.h"
#include "cv.h"
#include "highgui.h"
#include <string>

using namespace cv;

int main(int argc, char **argv) {
  ros::init(argc, argv, "main_vision_node");
  ros::NodeHandle n;

  // open up the video
  VideoCapture stream(ros::package::getPath("vision") + "/data/buoy_backing_up.avi");

  if (!stream.isOpened()) {
    ROS_ERROR("Cannot connect a camera or file! Shuting down...");
    return -1;
  }

  // various stages of images used in processing
  Mat raw_image, edge_image, dst, thresh_image, thresh_color;

  // process frames at 15hz
  ros::Rate loop_rate(15);

  // variables
  int frame_count = 0,  // a frame count is needed to loop the video
      canny_low   = 90, // the rest are parameters to the algorithms...
      canny_high  = 133,
      blur_number = 6,
      hue_max     = 50,
      hue_min     = 20;

  // create the GUI
  namedWindow("edge image", 2);
  createTrackbar("canny low", "edge image", &canny_low, 200);
  createTrackbar("canny high", "edge image", &canny_high, 200);
  createTrackbar("blur size", "edge image", &blur_number, 50);

  namedWindow("raw image", 2);
  namedWindow("HSV threshold", 2);
  createTrackbar("Hue min", "HSV threshold", &hue_min, 255);
  createTrackbar("Hue max", "HSV threshold", &hue_max, 255);

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

    // display images is seperate windows
    imshow("edge image", dst);
    imshow("raw image", raw_image);
    imshow("HSV threshold", thresh_color);

    // needed for the opencv GUI
    waitKey(1);

    // make sure we run at 15hz
    loop_rate.sleep();
  }

  destroyWindow("Image");
  return 0;
}
