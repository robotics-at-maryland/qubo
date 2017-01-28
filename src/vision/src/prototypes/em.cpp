#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace cv::ml;

int main(int argc, char*argv[]){
    
    //    if(argc != 2){
    //    cout << "wrong number of arguments dummy, call \"a.out test_video\"" << endl;
    //    return 0;
    // }
    cout << "OpenCV version : " << CV_VERSION << endl;
    cout << "Major version : " << CV_MAJOR_VERSION << endl;
    cout << "Minor version : " << CV_MINOR_VERSION << endl;
    cout << "Subminor version : " << CV_SUBMINOR_VERSION << endl;
    
    // Read image
    //VideoCapture cap = VideoCapture( argv[1] );
    //Mat im;
    //cap >> im;
    
    // Convert input image to HSV
    //Mat im_with_keypoints;
    
    //const TermCriteria termCrit(TermCriteria::COUNT+TermCriteria::EPS, EM::DEFAULT_MAX_ITERS, FLT_EPSILON);
    //EM test(1, (int)EM::COV_MAT_DIAGONAL, termCrit);
    
    //EM test();

    //CvEMParams params(DEFAULT_NCLUSTERS, EM::COV_MAT_DIAGONAL, const termCrit=TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, EM::DEFAULT_MAX_ITERS, 1e-6))
    
    //    while(!im.empty()){

    // waitKey(0);
    //   cap >> im;
    // }
}
