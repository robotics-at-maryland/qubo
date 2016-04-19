#include "ros/ros.h"
#include "ros/package.h"
#include "cv.h"
#include "highgui.h"
#include <opencv2/contrib/contrib.hpp>
#include <iostream>

using namespace cv;

int main(int argc, char **argv) {
  ros::init(argc, argv, "histogram_example");
  ros::NodeHandle n;

  // open up the video
  VideoCapture stream(ros::package::getPath("vision") + "/data/buoy_backing_up.avi");
  // VideoCapture stream(0);

  if (!stream.isOpened()) {
    ROS_ERROR("Cannot connect a camera or file! Shuting down...");
    return -1;
  }

  // various stages of images used in processing
  Mat raw_image, back_proj;
  Mat hist1, hist2, hist3;

  Mat target1 = imread("/home/michael/Documents/ram/bouy/buoy.png"),
      target2 = imread("/home/michael/Documents/ram/bouy/bouy2.png");
  Mat target_mask = imread("/home/michael/Documents/ram/bouy/buoy_mask.png"),
      target_mask2 = imread("/home/michael/Documents/ram/bouy/buoy_mask2.png");
  Mat targets[2] = {target1, target2};
  cvtColor(target_mask, target_mask, CV_BGR2GRAY);
  cvtColor(target_mask2, target_mask2, CV_BGR2GRAY);
  cvtColor(target1, target1, CV_BGR2HSV);
  cvtColor(target2, target2, CV_BGR2HSV);

  const int channels[] = {0 , 1};
  const int sizes[] = {180, 256};
  float range1[2] = {0, 180};
  float range2[2] = {0, 256};
  const float *ranges[] = {range1, range2};

  calcHist(&target1, 1, channels, target_mask, hist1, 2, sizes, ranges, true, false);
  calcHist(&target2, 1, channels, target_mask2, hist2, 2, sizes, ranges, true, false);
  addWeighted(hist1, 0.5, hist2, 0.5, 0.0, hist3);

  normalize(hist3, hist3, 0, 255, NORM_MINMAX, -1, Mat());

  // process frames at 15hz
  ros::Rate loop_rate(15);

  // variables
  int frame_count = 0,  // a frame count is needed to loop the video
      blur_number = 6,  // the rest are parameters to the algorithms...
      hue_max     = 50,
      hue_min     = 20;


  namedWindow("raw image", 2);

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

    cvtColor(raw_image, raw_image, CV_BGR2HSV);

    // histogram backprojection
    calcBackProject(&raw_image, 1, channels, hist3, back_proj, ranges);
    GaussianBlur(back_proj, back_proj, Size(11, 11), 5);
    threshold(back_proj, back_proj, 3, 255, CV_THRESH_BINARY);

    Mat image;
    cvtColor(raw_image, raw_image, CV_HSV2BGR);
    raw_image.copyTo(image, back_proj);

    // display images is seperate windows
    imshow("image", image);
    imshow("raw_image", raw_image);
    // needed for the opencv GUI
    waitKey(1);



    // make sure we run at 15hz
    loop_rate.sleep();
  }

  destroyWindow("Image");
  return 0;
}
