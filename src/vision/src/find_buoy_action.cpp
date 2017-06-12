#include "find_buoy_action.h"

using namespace cv;
using namespace std;


FindBuoyAction::FindBuoyAction(actionlib::SimpleActionServer<ram_msgs::VisionExampleAction> *as){

	namedWindow( "Gray image", CV_WINDOW_AUTOSIZE );
	
	//initialize background subtractors, keeping both in for now
	//m_pMOG = bgsegm::createBackgroundSubtractorMOG(1000,5,.7,0);
	m_as = as; 
	m_pMOG = createBackgroundSubtractorMOG2(10000, 35, false);


	// Setup SimpleBlobDetector parameters.
	SimpleBlobDetector::Params params;
	
	// Change thresholds
	params.minThreshold = 0;
	params.maxThreshold = 256;

	//Filter by Area
	params.filterByArea = true;
	params.minArea = 100;

	// Set up detector with params
	m_detector = SimpleBlobDetector::create(params);
	

}

FindBuoyAction::~FindBuoyAction(){}
	


//TODO, possibly pass in the matrix we want the data copied into? seems more consistent with what OpenCV does
Mat FindBuoyAction::backgroundSubtract(Mat cframe){
	
    Mat local_frame; //local meaning within the scope of the function

    //update the background model
	m_pMOG->apply(cframe, local_frame);
	
    //blurs the image uses the MOG background subtraction
    GaussianBlur(local_frame, local_frame, Size(3,3), 0,0);

    // Define the structuring elements to be used in eroding and dilating the image 
    Mat se1 = getStructuringElement(MORPH_RECT, Size(5, 5));
    Mat se2 = getStructuringElement(MORPH_RECT, Size(5, 5));

    // Perform dialting and eroding helps to elminate background noise
	//sgillen@20175109-15:51 when I found this on the second operation was actually being performed
    morphologyEx(local_frame, local_frame, MORPH_CLOSE, se1);
    morphologyEx(local_frame, local_frame, MORPH_OPEN, se2);

    //inverts the colors 
    bitwise_not(local_frame, local_frame, noArray()); 
	
    return local_frame;   
}


void FindBuoyAction::updateAction(Mat cframe) {   
	//create Background Subtractor objects
    
    Mat mog_output; 
	vector<KeyPoint> keypoints; // Storage for blobs

	ram_msgs::VisionExampleFeedback feedback;
	Point2f center; 
	
	mog_output = backgroundSubtract(cframe); //updates the MOG frame
    m_detector->detect(mog_output, keypoints);
	
	
	for (auto& point:keypoints ){
		ROS_ERROR("%f %f" , point.pt.x, point.pt.y); 
	}
	
	imshow("Gray image" , mog_output);
	waitKey(30);

	ROS_ERROR("let's see if we see something");
	if (updateHistory(mog_output, keypoints, center)){
		feedback.x_offset = cframe.rows/2 - center.x; 
		feedback.y_offset = cframe.cols/2 - center.y;

		ROS_ERROR("publishing feedback");
		//I actually think it might be better to keep the action server away from this class, haven't decided yet..
		m_as->publishFeedback(feedback);
	}

}


//Davids way of keeping track of the history of points
//TODO, should probably pass in 
bool  FindBuoyAction::updateHistory(Mat cframe, vector<KeyPoint> keypoints, Point2f center){
    float pointX, pointY, x, y; 
    bool insert; 
    Vec3b color;
    int age = 10, offSet = 20, filter = 30, offSet2 = 5;//how long ago the blob was first seen and the offset of the center and the value for the color we want to see 

    //for every deteced blob either add it if its new or update current one 
    for (auto& point:keypoints ){    
        color = cframe.at<Vec3b>(point.pt); 
        cout << color << endl;
        insert = false;
        pointX = point.pt.x;
        pointY = point.pt.y;
        for (std::vector<tuple< Point2f, Vec3b, int >>::iterator it = m_history.begin(); it != m_history.end(); it++){ 
            x = std::get<0>(*it).x;
            y = std::get<0>(*it).y;   

            //if blob is within offSet pixels of a know blob update the blob to the new blobs location       
            if (((pointX <= x + offSet) && (pointX >= x - offSet)) && ((pointY <= y + offSet && (pointY >= y - offSet)))){
                m_history.erase(it);
                m_history.emplace_back(std::make_tuple (point.pt,color,0));
                insert = true;
            }
            std::get<2>(*it) += 1;
            //if the blobs hasnt been updated in age frames remove it 
            if (std::get<2>(*it) > age){
                m_history.erase(it);
            }
        }
        if (!insert)
            m_history.emplace_back(std::make_tuple (point.pt, color, 0));
    }
    //outputs the buoys offset if it is the right color.  
    for (std::vector<tuple< Point2f, Vec3b, int >>::iterator it = m_history.begin(); it != m_history.end(); it++){
        color = std::get<1>(*it);
        if (color[0] >= filter - offSet2 && color[0] <= filter + offSet2){
            center =  std::get<0>(*it);
            return true;}
    } 

    return false;
}
