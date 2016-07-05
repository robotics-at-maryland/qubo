#ifndef VISION_CORE_H
#define VISION_CORE_H

#include "ros/ros.h"
#include "image_transport/image_transport.h"

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "cv_bridge/cv_bridge.h"

#include "sensor_msgs/image_encodings.h"
#include "vector"

class vision_node {
    
    public:
    cv::Mat raw_image, 
    edge_image, 
    dst, 
    thresh_image, 
    thresh_color, 
    gray_image, 
    keypoint_image;
 
  int frame_count = 0,  // a frame count is needed to loop the video
      canny_low   = 90, // the rest are parameters to the algorithms...
      canny_high  = 133,
      blur_number = 6,
      hue_max     = 50,
      hue_min     = 20;

  
    vision_node(int, char **, int);
    ~vision_node();

    void update();

    private:

    void imageCallback(const sensor_msgs::ImageConstPtr &msg);
    ros::NodeHandle nh;
    image_transport::Subscriber camera_sub;
};


#endif
