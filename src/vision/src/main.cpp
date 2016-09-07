#include <stdio.h>
#include <opencv2/opencv.hpp>

//TODO find some standardized way to keep videos in our repo, maybe just commit them? does github have a size limit?
#define TEST_FEED "/home/sgillen/not_bouy.avi"


int main(void){

    
    //we'll have to be careful about this, may be best to have a vehicle layer node pass this object to vision, or maybe even just pass the filepath? 
    //and do we want to support mono vision in the future? can we build a system that will allow that?
    //cv::VideoCapture capture(/dev/cam0)
    cv::VideoCapture capture(TEST_FEED);

    //Capturing a frame:
    if(!capture.isOpened()){            // capture a frame 
        printf("couldn't open file/camera\n");
        exit(0);
    }
    

        cv::Mat img;
        capture >> img;
        std::cout << img;
   
}
