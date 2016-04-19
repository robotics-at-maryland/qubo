#include "ros/ros.h"
#include "ros/package.h"
#include "cv.h"
#include "highgui.h"
#include <iostream>
#include <cmath>
#include <algorithm>

using namespace cv;


int main(int argc, char **argv) {
  ros::init(argc, argv, "adaptive_threshold_example");
  ros::NodeHandle n;

  // open up the video
  VideoCapture stream("/home/michael/Documents/ram/data/misc/20100715100703.avi");
  // VideoCapture stream(0);

  if (!stream.isOpened()) {
    ROS_ERROR("Cannot connect a camera or file! Shuting down...");
    return -1;
  }

  // various stages of images used in processing
  Mat raw_image, thresh_image, thresh_color;

  // blob detector
  SimpleBlobDetector::Params params;
  // params.blobColor = 255;
  params.minDistBetweenBlobs = 1.0f;
  params.filterByInertia = false;
  params.filterByConvexity = true;
  params.minConvexity = 0.8;
  params.filterByColor = false;
  params.filterByCircularity = true;
  params.minCircularity = 0.7;
  params.filterByArea = true;
  params.minArea = 500.0f;
  params.maxArea = 10000.0f;
  SimpleBlobDetector blob_detector(params);
  std::vector<KeyPoint> keypoints;
  std::vector<std::vector<cv::Point> > contours;
  std::vector<cv::Vec4i> hierarchy;

  // process frames at 15hz
  ros::Rate loop_rate(15);

  // variables
  int frame_count = 0,  // a frame count is needed to loop the video
      blur_number = 6,  // the rest are parameters to the algorithms...
      hue_max     = 48,
      hue_min     = 11,
      kernel_size = 1;


  namedWindow("raw image", 2);
  namedWindow("HSV threshold", 2);
  createTrackbar("Hue min", "HSV threshold", &hue_min, 255);
  createTrackbar("Hue max", "HSV threshold", &hue_max, 255);
  createTrackbar("Kernal size", "HSV threshold", &kernel_size, 10);

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
    cvtColor(raw_image, thresh_image, CV_BGR2HSV);
    medianBlur(thresh_image, thresh_image, pow(2, kernel_size) + 1);
    inRange(thresh_image, Scalar(hue_min, 94, 118), Scalar(hue_max, 255, 255), thresh_image);
    Canny(thresh_image, thresh_color, 100, 200, 3);

    // find contours (if always so easy to segment as your image, you could just add the black/rect pixels to a vector)

    cv::findContours(thresh_image, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

    /// Draw contours and find biggest contour (if there are other contours in the image, we assume the biggest one is the desired rect)
    // drawing here is only for demonstration!
    int biggestContourIdx = -1;
    float biggestContourArea = 0;
    // cv::Mat drawing = cv::Mat::zeros(mask.size(), CV_8UC3);

    for( int i = 0; i< contours.size(); i++ ) {
      cv::Scalar color = cv::Scalar(0, 0, 255);
      float ctArea= cv::contourArea(contours[i]);
      if (ctArea < 1000) continue;

      vector<Point> approx;
      approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true)*0.02, true);
      contours[i] = approx;
      if (approx.size() != 4) continue;

      // find center point
      float x = 0, y = 0;
      for (int j = 0; j < 4; j++) {
        x += approx[j].x;
        y += approx[j].y;
      }
      x /= 4;
      y /= 4;

      // find direction
      float dists[4] = {0};
      for (int j = 1; j < 4; j++) {
        dists[j] = sqrt(pow(approx[0].x - approx[j].x, 2) +
                        pow(approx[0].y - approx[j].y, 2));
      }



      circle(raw_image, Point(x, y), 5, color);


      drawContours(raw_image, contours, i, color, 5, 8, hierarchy, 0, cv::Point());


      std::cout << approx.size() << std::endl;
      if(ctArea > biggestContourArea) {
          biggestContourArea = ctArea;
          biggestContourIdx = i;
      }
    }
    // thresh_color = Scalar::all(0);
    // raw_image.copyTo(thresh_color, thresh_image);

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
