#include "find_buoy_action.h"

using namespace cv;
using namespace std;


Mat FindBuoyAction::backgroundSubtract(Mat cframe){
    Mat gauss, mask, invert;

    //update the background model
    pMOG2->apply(frame, fgMaskMOG2);
    pMOG->apply(frame, fgMaskMOG);

    //blurs the image uses the MOG background subtraction
    GaussianBlur(fgMaskMOG2, gauss, Size(3,3), 0,0);

    // Define the structuring elements to be used in eroding and dilating the image 
    Mat se1 = getStructuringElement(MORPH_RECT, Size(5, 5));
    Mat se2 = getStructuringElement(MORPH_RECT, Size(5, 5));

    // Perform dialting and eroding helps to elminate background noise 
    morphologyEx(gauss, mask, MORPH_CLOSE, se1);
    morphologyEx(gauss, mask, MORPH_OPEN, se2);

    //inverts the colors 
    bitwise_not(mask, invert, noArray()); 
	
    return invert;   
}

vector<KeyPoint> FindBuoyAction::detectBuoy(Mat cframe){
	// Setup SimpleBlobDetector parameters.
	SimpleBlobDetector::Params params;
	 
	// Change thresholds
	params.minThreshold = 0;
	params.maxThreshold = 256;

	//Filter by Area
	params.filterByArea = true;
	params.minArea = 425;

	// Storage for blobs        
	vector<KeyPoint> keypoints;

	// Set up detector with params
	Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);

	// SimpleBlobDetector::create creates a smart pointer. 
	// So you need to use arrow ( ->) instead of dot ( . )
	detector->detect(cframe, keypoints);

	return keypoints;	
}


float* FindBuoyAction::processVideo(VideoCapture capture, Server *as) {   
	//create Background Subtractor objects
    pMOG = bgsegm::createBackgroundSubtractorMOG(1000,5,.7,0);
    pMOG2 = createBackgroundSubtractorMOG2(10000, 35, false);

    Mat copy, thresh, invert;
    // Storage for blobs        
	vector<KeyPoint> keypoints;

	float out[2];

    if (!capture.isOpened()){
        //error in opening the video input
        // cerr << "Unable to open video file: " << videoFilename << endl;
		ROS_ERROR("error opening video, exiting");
        exit(EXIT_FAILURE);
    }

	ROS_ERROR("opened video");
    // VideoWriter outputVideo; //output video object
    // Size S = Size((int) capture.get(CV_CAP_PROP_FRAME_WIDTH),    //Acquire input size
    //               (int) capture.get(CV_CAP_PROP_FRAME_HEIGHT));

    // int ex = static_cast<int>(capture.get(CV_CAP_PROP_FOURCC));     // Get Codec Type- Int form
    // //outputVideo.open("/home/dlinko/Desktop/log.avi", ex , capture.get(CV_CAP_PROP_FPS),S,true);

    // if(!outputVideo.isOpened()){
    //     cout << "something went wrong with opening the output video" << endl;
    //     cout << ex << endl;
    //     cout << S << endl;        
    //     cout << capture.get(CV_CAP_PROP_FPS) << endl;
    // }

    //read input data. ESC or 'q' for quitting
    while ((char)keyboard != 'q' && (char)keyboard != 27 ){
         
        //read the current frame
        if (!capture.read(frame)) {
            cerr << "Unable to read next frame." << endl;
            cerr << "Exiting..." << endl;
            exit(EXIT_FAILURE);
        }

		ram_msgs::VisionExampleFeedback feedback;

        //get the frame number and write it on the current frame
        stringstream ss;
        rectangle(frame, cv::Point(10, 2), cv::Point(100,20),
                  cv::Scalar(255,255,255), -1);
        ss << capture.get(CAP_PROP_POS_FRAMES);
        string frameNumberString = ss.str();
        putText(frame, frameNumberString.c_str(), cv::Point(15, 15),
                FONT_HERSHEY_SIMPLEX, 0.5 , cv::Scalar(0,0,0));

        invert = backgroundSubtract(frame);
        keypoints = detectBuoy(invert);
      	   
        // Draw detected blobs as red circles.
        // DrawMatchesFlags::DRAW_RICH_KEYPOINTS flag ensures
        // the size of the circle corresponds to the size of blob

        // Mat im_with_keypoints;
        // drawKeypoints(frame, keypoints, im_with_keypoints, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
        
        bool find;
        cvtColor(frame,frame,COLOR_BGR2HSV); 
        find = updateHistory(keypoints);

		//sgillen@20170507-15:05 you can use an exception here if you want, minor though
        if (find){
        	float x_offset = capture.get(CV_CAP_PROP_FRAME_WIDTH)/2; 
    		float y_offset = capture.get(CV_CAP_PROP_FRAME_HEIGHT)/2;
    		//capture.release();
    		feedback.x_offset = (x_offset - center.x); //sgillen@20173507-15:35 - feel like these variable names are confusing
    		feedback.y_offset = (y_offset - center.y);
			as->publishFeedback(feedback);

			
        }

        //outputVideo.write(im_with_keypoints);

        //get the input from the keyboard
        //keyboard = waitKey(30);
    }
    //delete capture object
    capture.release();
    out[0] = -1;  
    out[1] = -1;
    return out;  //sgillen@20172007-15:20 - Danger!! out is a local c array (which is just a pointer to a place in memory). This pointer is pointing to memory
	//in the function stack frame, as soon as you return that memory is freed (or is is to be freed anyway), anyone who tries to return out will likely segfault immediately 

}

bool  FindBuoyAction::updateHistory(vector<KeyPoint> keypoints){
    float pointX, pointY, x, y; 
    bool insert; 
    Vec3b color;
    int age = 10, offSet = 20, filter = 30, offSet2 = 5;//how long ago the blob was first seen and the offset of the center and the value for the color we want to see 

    //for every deteced blob either add it if its new or update current one 
    for (auto& point:keypoints ){    
        color = frame.at<Vec3b>(point.pt); 
        cout << color << endl;
        insert = false;
        pointX = point.pt.x;
        pointY = point.pt.y;
        for (std::vector<tuple< Point2f, Vec3b, int >>::iterator it = history.begin(); it != history.end(); it++){ 
            x = std::get<0>(*it).x;
            y = std::get<0>(*it).y;   

            //if blob is within offSet pixels of a know blob update the blob to the new blobs location       
            if (((pointX <= x + offSet) && (pointX >= x - offSet)) && ((pointY <= y + offSet && (pointY >= y - offSet)))){
                history.erase(it);
                history.emplace_back(std::make_tuple (point.pt,color,0));
                insert = true;
            }
            std::get<2>(*it) += 1;
            //if the blobs hasnt been updated in age frames remove it 
            if (std::get<2>(*it) > age){
                history.erase(it);
            }
        }
        if (!insert)
            history.emplace_back(std::make_tuple (point.pt, color, 0));
    }
    //outputs the buoys offset if it is the right color.  
    for (std::vector<tuple< Point2f, Vec3b, int >>::iterator it = history.begin(); it != history.end(); it++){
        color = std::get<1>(*it);
        if (color[0] >= filter - offSet2 && color[0] <= filter + offSet2){
            center =  std::get<0>(*it);
            return true;}
    } 

    return false;
}
