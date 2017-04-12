#include <stdio.h>
#include <opencv2/opencv.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/video.hpp>

#include "ros/ros.h"
#include <iostream>

#include <ram_msgs/VisionExampleAction.h>
#include <actionlib/server/simple_action_server.h>

#include "std_msgs/String.h"
#include  "ram_msgs/bool_bool.h"


typedef actionlib::SimpleActionServer<ram_msgs::VisionExampleAction> Server;


class VisionNode{

    public:

    //you need to pass in a node handle and a camera feed, which should be a file path either to a physical device or to a video file
    VisionNode(std::shared_ptr<ros::NodeHandle> n, std::string feed);
    ~VisionNode();
    void update(); //this will just pull the next image in


    //all service prototypes should go below, you also need to add a service variable for it in here and actually register
    //it in the constructor
    //=================================================================================================================


    bool service_test(ram_msgs::bool_bool::Request &req, ram_msgs::bool_bool::Response &res);

    bool buoy_detector(ram_msgs::bool_bool::Request &req, ram_msgs::bool_bool::Response &res);


    
    //sg: put action definitions here
    //=================================================================================================================

    static void test_execute(const ram_msgs::VisionExampleGoalConstPtr& goal, Server*as);


    
    protected:
    

    std::shared_ptr<ros::NodeHandle> n;
    //cap is the object holding the video feed, either real or from an existing avi file
    //img is the object reprenting the current image we're looking at, we'll keep pumping the next fram
    //from cap into img at every update

    //cap is a video capture object, img is a Mat object that gets updated every time step
    cv::VideoCapture cap;
    cv::Mat img;

    //declare a service object for your service below
    //======================================================================
    ros::ServiceServer buoy_detect_srv;
    ros::ServiceServer test_srv;

    
    //declare an action server object for your action here
    //======================================================================
    //the VisionExampleAction name here comes from the .action file in qubo/ram_msgs/action.
    //the build system appends the word Action to whatever the file name is in the ram_msgs directory
    actionlib::SimpleActionServer<ram_msgs::VisionExampleAction> example_server;
  
   
};
