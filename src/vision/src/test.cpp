#include <stdio.h>
#include <opencv2/opencv.hpp>

int main(void){
    cv::VideoCapture capture("/home/sgillen/not_bouy.avi");
    //Capturing a frame:
    if(!capture.isOpened()){            // capture a frame 
        printf("couldn't open file/camera\n");
        exit(0);
    }

    cv::Mat img;

    capture >> img;
    std::cout << img;
    
}
