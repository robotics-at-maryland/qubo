#ifndef FIND_BUOY_TUNER_H
#define FIND_BUOY_TUNER_H


#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include <opencv2/opencv.hpp>
#include "opencv2/bgsegm.hpp"

//C
#include <stdio.h>

//C++
#include <iostream>
#include <sstream>
#include <tuple>        // std::tuple, std::get, std::tie, std::ignore
#include <vector>

#include "vision_node.h"


class BuoyActionTuner{
    public:

    BuoyActionTuner(actionlib::SimpleActionServer<ram_msgs::VisionExampleAction> *as, cv::VideoCapture cap);
    ~BuoyActionTuner();

    cv::Mat backgroundSubtract(cv::Mat cframe);
    std::vector<cv::KeyPoint> detectBuoy(cv::Mat cframe);
	void updateAction(const cv::Mat);
	bool updateHistory(cv::Mat cframe, std::vector<cv::KeyPoint> keypoints, cv::Point2f center);
    
    protected:

    cv::VideoCapture m_cap;

    static int m_slider_area, m_slider_circularity, m_slider_convexity, m_slider_ratio;
    
	actionlib::SimpleActionServer<ram_msgs::VisionExampleAction> *m_as;

	
    
	ram_msgs::VisionExampleFeedback m_feedback;

    static cv::SimpleBlobDetector::Params m_params;
    static cv::Ptr<cv::SimpleBlobDetector> m_detector;

    static cv::Ptr<cv::BackgroundSubtractor> m_pMOG;//MOG Background subtractor
    
    std::vector<std::tuple<cv::Point2f, cv::Vec3b, int>> m_history;
    
    //track bar callbacks
    static void areaCallback( int, void* );
    static void circCallback( int, void* );
    static void convCallback( int, void* );
    static void inertiaCallback( int, void* );
	
};

#endif
