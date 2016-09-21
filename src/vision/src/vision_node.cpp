#include "vision_node.h"

VisionNode::VisionNode(std::shared_ptr<ros::NodeHandle> n, int rate, std::string feed){

    //we'll have to be careful about this, may be best to have a vehicle layer node pass this object to vision, or maybe even just pass the filepath? 
    //and do we want to support mono vision in the future? can we build a system that will allow that?
    //cv::VideoCapture capture(/dev/cam0)
    cv::VideoCapture capture(feed);

    //Capturing a frame:
    if(!capture.isOpened()){            // capture a frame 
        printf("couldn't open file/camera\n");
        exit(0);
    }

}    

VisionNode::~VisionNode(){}

void VisionNode::update(){
    cap >> img;
    //std::cout << img;
}

bool VisionNode::buoy_detector(ram_msgs::bool_bool::Request &req, ram_msgs::bool_bool::Response &res){
    
    // code goes here
   

    return 0;

}
