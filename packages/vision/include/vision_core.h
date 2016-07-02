#ifndef VISION_CORE_H
#define VISION_CORE_H

#include "ros/ros.h"
#include "image_transport/image_transport.h"
#include "opencv2/highgui/highgui.hpp"
#include "cv_bridge/cv_bridge.h"

//#include <image_transport/image_transport.h>
//#include <opencv2/highgui/highgui.hpp>
//#include <cv_bridge/cv_bridge.h>
//#include <opencv2/contrib/contrib.hpp>
//#include <opencv2/core/core.hpp>
//#include <vector>

class vision_node {
    
    public:
    //! Constructor.
    vision_node(int, char **, int);
    ~vision_node();

    void update();

    private:
    cv::Mat raw_image; 
    cv::Mat edge_image; 
    cv::Mat dst; 
    cv::Mat thresh_image; 
    cv::Mat thresh_color; 
    cv::Mat gray_image; 
    cv::Mat keypoint_image;

     int frame_count = 0,
	  canny_low   = 90,
          canny_high  = 133,
          blur_number = 6,
          hue_max     = 50,
          hue_min     = 20;

   
    std::vector<cv::KeyPoint> orb_key_points;
    //! Callback function for subscriber.
    */
    void imageCallback(const sensor_msgs::ImageConstPtr &msg);
    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);
    image_transport::Subscriber sub;
};


#endif
