#ifndef BLOB_ACTION_H
#define BLOB_ACTION_H


//C++
#include <iostream>
#include <sstream>
#include <tuple>        // std::tuple, std::get, std::tie, std::ignore
#include <vector>

#include "vision_node.h"


class BlobAction{
    public:
    BlobAction(actionlib::SimpleActionServer<ram_msgs::VisionNavAction> *as);
    ~BlobAction();

    void updateAction(const cv::Mat cframe);
    std::vector<cv::KeyPoint> detectBuoy(cv::Mat cframe);
    protected:


	actionlib::SimpleActionServer<ram_msgs::VisionNavAction> *m_as;
    cv::Ptr<cv::SimpleBlobDetector> m_detector;
	ram_msgs::VisionNavFeedback m_feedback;

	
};

#endif
