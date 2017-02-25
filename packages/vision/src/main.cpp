#include "ros/ros.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>

using namespace std;
using namespace cv;


int main(int argc, char **argv) {


  ros::init(argc, argv, "main_vision_node");

  ros::NodeHandle n;

//  Mat img = imread("image.jpg", CV_LOAD_IMAGE_UNCHANGED); //read the image data in the file "MyPic.JPG" and store it in 'img'

  //  if (img.empty()) //check whether the image is loaded or not
  //  {
  //       cout << "Error : Image cannot be loaded..!!" << endl;
  //       //system("pause"); //wait for a key press
  //       return -1;
  //  }

  VideoCapture stream(CV_CAP_ANY);

  if (!stream.isOpened()) {
    ROS_ERROR("Cannot connect to a camera!! Shuting down...");
    return -1;
  }
  Mat raw_image, edge_image, dst;


  namedWindow("Image", CV_WINDOW_AUTOSIZE); //create a window with the name "MyWindow"
   //imshow("MyWindow", img); //display the image which is stored in the 'img' in the "MyWindow" window


   //destroy the window with the name, "MyWindow"

  ros::Rate loop_rate(60);

  while (ros::ok()) {
    //stream.read(raw_image);
    raw_image = imread("/home/michael/src/ram/src/qubo/packages/vision/src/targets.jpg");

    if (raw_image.empty()) {
      ROS_ERROR("Cannot load image");
      return -1;
    }
    cvtColor(raw_image, edge_image, CV_BGR2GRAY);
    blur(edge_image, edge_image, Size(3, 3));
    Canny(edge_image, edge_image, 20, 60, 3);
    dst = Scalar::all(0);

    raw_image.copyTo(dst, edge_image);

    imshow("Image", edge_image);
    waitKey(1);

    loop_rate.sleep();
  }
  destroyWindow("Image");
  return 0;
}
