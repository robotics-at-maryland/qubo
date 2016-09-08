#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "ros/ros.h"
#include <iostream>
#include "std_msgs/String.h"


class VisionNode{

    public:
    VisionNode(std::shared_ptr<ros::NodeHandle>, int rate, std::string feed);
    ~VisionNode();
    void update(); //this will just pull the next image in
    
    protected:
    
    //cap is the object holding the video feed, either real or from an existing avi file    
    //img is the object reprenting the current image we're looking at, we'll keep pumping the next fram
    //from cap into img at every update

    //sg: We'll need three of these for our three cameras, I'll need to think of a good general way 
    //of doing that without tying us too much to the current configuration, I'll leave th
    cv::VideoCapture cap;    
    cv::Mat img;


};
