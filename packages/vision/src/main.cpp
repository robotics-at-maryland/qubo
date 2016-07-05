<<<<<<< HEAD
#include "vision_core.h"
=======
#include "ros/ros.h"
#include "ros/package.h"
#include "cv.h"
#include "highgui.h"
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/core/core.hpp>
#include <vector>
#include "opencv2/opencv.hpp"
>>>>>>> 7601d10797bbb4c52b3ac2903bd318493c5b2ea6


int main(int argc, char **argv) {  
    ros::init(argc, argv, "vision_node");    
    vision_node node(argc, argv, 10);

    cv::namedWindow("edge image", 2);
    cv::createTrackbar("canny low", "edge image", &node.canny_low, 200);
    cv::createTrackbar("canny high", "edge image", &node.canny_high, 200);
    cv::createTrackbar("blur size", "edge image", &node.blur_number, 50);

    cv::namedWindow("raw image", 2);
    cv::namedWindow("HSV threshold", 2);
    cv::createTrackbar("Hue min", "HSV threshold", &node.hue_min, 255);
    cv::createTrackbar("Hue max", "HSV threshold", &node.hue_max, 255);


    while (ros::ok()){
        node.update();    
    }
    
    return 0;
}

