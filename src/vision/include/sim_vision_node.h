#ifndef SIM_VISION_NODE_H
#define SIM_VISION_NODE_H

//opencv includes
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/video.hpp>

#include <iostream>
#include <stdio.h>

//ros includes
#include "ros/ros.h"
#include <ram_msgs/VisionExampleAction.h>
#include <actionlib/server/simple_action_server.h>

//message includes
#include "std_msgs/String.h"
#include  "ram_msgs/bool_bool.h"

//cv_bridge/image transport includes
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>

//our actions
#include "buoy_action.h"
#include "gate_action.h"

class SimVisionNode{

    public:

    
    image_transport::ImageTransport m_it;
    image_transport::Subscriber m_image_sub;
    void imageCallback(const sensor_msgs::ImageConstPtr& msg);
    
    //you need to pass in a node handle and a camera feed, which should be a file path either to a physical device or to a video file
    SimVisionNode(ros::NodeHandle n, ros::NodeHandle np,  std::string feed);
    ~SimVisionNode();
    void update(); //this will just pull the next image in


    //all service prototypes should go below, you also need to add a service variable for it in here and actually register
    //it in the constructor
    //=================================================================================================================


    bool serviceTest(ram_msgs::bool_bool::Request &req, ram_msgs::bool_bool::Response &res);
    
    
    //sg: put action definitions here
    //=================================================================================================================

    static void testExecute(const ram_msgs::VisionExampleGoalConstPtr& goal, actionlib::SimpleActionServer<ram_msgs::VisionExampleAction>*as);

    void findBuoy(const ram_msgs::VisionExampleGoalConstPtr& goal, actionlib::SimpleActionServer<ram_msgs::VisionExampleAction> *as);
    void findGate(const ram_msgs::VisionExampleGoalConstPtr& goal, actionlib::SimpleActionServer<ram_msgs::VisionExampleAction> *as);
    
    protected:

    
    cv::Mat m_img;
	
    //declare a service object for your service below
    //======================================================================
    ros::ServiceServer m_test_srv;

    
    //declare an action server object for your action here
    //======================================================================
    //the VisionExampleAction name here comes from the .action file in qubo/ram_msgs/action.
    //the build system appends the word Action to whatever the file name is in the ram_msgs directory
    actionlib::SimpleActionServer<ram_msgs::VisionExampleAction> m_buoy_server;
	actionlib::SimpleActionServer<ram_msgs::VisionExampleAction> m_gate_server;
};


#endif
