#ifndef BUOY_ACTION_H
#define BUOY_ACTION_H


/* #include "opencv2/imgcodecs.hpp" */
//#include "opencv2/imgproc.hpp"
/* #include "opencv2/videoio.hpp" */
//#include <opencv2/highgui.hpp>
//#include <opencv2/video.hpp>
//#include <opencv2/opencv.hpp>
/* #include "opencv2/bgsegm.hpp" */
//#include <opencv2/features2d.hpp>
//C
#include <stdio.h>

//C++
#include <iostream>
#include <sstream>
#include <tuple>        // std::tuple, std::get, std::tie, std::ignore
#include <vector>

#include "vision_node.h"


class BuoyAction{
    public:
    BuoyAction(actionlib::SimpleActionServer<ram_msgs::VisionNavAction> *as);
    ~BuoyAction();

    cv::Mat backgroundSubtract(cv::Mat cframe);
    std::vector<cv::KeyPoint> detectBuoy(cv::Mat cframe);
	void updateAction(cv::Mat cframe);
	bool updateHistory(cv::Mat cframe, std::vector<cv::KeyPoint> keypoints, cv::Point2f center);
    
    protected:


	actionlib::SimpleActionServer<ram_msgs::VisionNavAction> *m_as;
	cv::Ptr<cv::SimpleBlobDetector> m_detector;
	cv::Ptr<cv::BackgroundSubtractor> m_pMOG;//MOG Background subtractor
    std::vector<std::tuple<cv::Point2f, cv::Vec3b, int>> m_history;
	ram_msgs::VisionNavFeedback m_feedback;

	
};

#endif
