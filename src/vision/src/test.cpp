#include <stdio.h>
#include <opencv2/opencv.hpp>



int main(void){
    CvCapture* capture = cvCaptureFromAVI("infile.avi");
    //Capturing a frame:
    printf("wow we didn't crash horribly (well not yet anyway)\n");
    IplImage* img = 0; 
    if(!cvGrabFrame(capture)){              // capture a frame 
        printf("Could not grab a frame\n\7");
        exit(0);
    }

    img=cvRetrieveFrame(capture);// retrieve the captured frame

}
