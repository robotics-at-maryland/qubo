//sg: The intention of this was to fit a single guassian to each color plane of a buoy image.
//

#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main(int argc, char*argv[]){
    
    if(argc != 2){
        cout << "wrong number of arguments dummy, call \"a.out test_video\"" << endl;
        return 0;
    }
    

    // Read image
    Mat im;
    VideoCapture cap = VideoCapture( argv[1] );
    cap >> im;

    Mat planes[3];
    split(im, planes);
    
    //    imshow("plane1" , planes[0]);
    //imshow("plane2" , planes[1]);
    // imshow("plane3" , planes[2]);
        
    while(!im.empty()){
        Mat means;
        Mat stdev;
        
        meanStdDev(im, means, stdev, noArray());
        cout << means;
        cout << stdev;
        imshow("image", im);
        
        cin.ignore();
    }

}
