#include "ros/ros.h"
#include "ros/package.h"
#include "cv.h"
#include "highgui.h"
#include <iostream>
#include <cmath>
#include <algorithm>

using namespace cv;


/* Used to store pipe detections */
class PipeDetection {
public:
  RotatedRect rect;
  float angle;

  PipeDetection(RotatedRect rect, float angle) {
    this->rect = rect;
    this->angle  = angle;
  }
};


void detectPipes(vector< vector<Point> > *contours, vector<PipeDetection> *result) {
  for (int i = 0; i < contours->size(); i++) {
    // filter on contour area
    float ctArea = cv::contourArea((*contours)[i]);
    if (ctArea < 1000) continue;

    // fit rect
    RotatedRect min_rect = minAreaRect(Mat((*contours)[i]));

    // get an angle between 0 and 180
    float angle = min_rect.size.width < min_rect.size.height
                    ? min_rect.angle + 180
                    : min_rect.angle + 90;

    result->push_back(PipeDetection(min_rect, angle));
  }
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "adaptive_threshold_example");
  ros::NodeHandle n;

  // open up the video
  //VideoCapture stream("/home/michael/Documents/ram/data/misc/20100715100703.avi");
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
      hue_max     = 48,
      hue_min     = 11,
      kernel_size = 1;

  // used for drawing
  cv::Scalar color = cv::Scalar(0, 0, 255);

  vector< vector<Point> > contours;
  vector<PipeDetection> dets;


  namedWindow("raw image", 2);
  namedWindow("HSV threshold", 2);
  createTrackbar("Hue min", "HSV threshold", &hue_min, 255);
  createTrackbar("Hue max", "HSV threshold", &hue_max, 255);
  createTrackbar("Kernal size", "HSV threshold", &kernel_size, 10);

  // main loop
  while (ros::ok()) {
    contours.clear();
    dets.clear();

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
    cvtColor(raw_image, thresh_image, CV_BGR2HSV);
    medianBlur(thresh_image, thresh_image, pow(2, kernel_size) + 1);
    inRange(thresh_image, Scalar(hue_min, 94, 118), Scalar(hue_max, 255, 255), thresh_image);
    Canny(thresh_image, thresh_color, 100, 200, 3);

    // find contours
    cv::findContours(thresh_image, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

    // detect the pipes
    detectPipes(&contours, &dets);

    // draw the pipes
    for (int j=0; j<dets.size(); j++) {
      circle(raw_image, dets[j].rect.center, 5, color);
      Point2f rect_points[4];
      dets[j].rect.points(rect_points);
      for (int j = 0; j < 4; j++)
        line(raw_image, rect_points[j], rect_points[(j+1)%4], color, 1, 8);
    }

    // display images is seperate windows
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
