#ifndef VISION_CORE_H
#define VISION_CORE_H

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>

class vision_node {
    
    public:
    //! Constructor.
    vision_node(std::shared_ptr<ros::NodeHandle>  , int);
		
    void update();
    //! Destructor.
    ~vision_node();


    private:
    Mat raw_image = cv::Scalar::all(0); 
    cv::Mat edge_image = cv::Scalar::all(0), 
    cv::Mat dst = cv::Scalar::all(0), 
    cv::Mat thresh_image = cv::Scalar::all(0), 
    cv::Mat thresh_color = cv::Scalar::all(0), 
    cv::Mat gray_image = cv::Scalar::all(0), 
    cv::Mat keypoint_image cv::Scalar::all(0);


     int frame_count = 0,  // a frame count is needed to loop the video
	  canny_low   = 90, // the rest are parameters to the algorithms...
          canny_high  = 133,
          blur_number = 6,
          hue_max     = 50,
          hue_min     = 20;

    ros::Rate loop_rate;

    std::vector<cv::KeyPoint> orb_key_points;
    //! Callback function for subscriber.
    void imageCallback(const sensor_msgs::ImageConstPtr &msg);
    ros::Subscriber camera_sub;


};


#endif
