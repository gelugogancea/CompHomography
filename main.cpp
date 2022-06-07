#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <time.h>
#include <mutex>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>


using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

const int MAX_FEATURES = 500;
const float GOOD_MATCH_PERCENT = 0.15f;

std::mutex mtx;
char *old_ctype;
Mat img;
Mat myrefimg;
Mat imReg, h;

void alignImages(Mat &im1, Mat &im2, Mat &im1Reg, Mat &h);
int mymain(Mat imReference,Mat im );

int main(int argc, char *argv[])
{
   
  VideoCapture cap(0);

 
  if(!cap.isOpened()){
    cout << "Error opening video stream or file" << endl;
    return -1;
  }

  while(1){

    cv::Mat frame;
    
    cap >> frame;

  
    if (frame.empty())
      break;
    mymain(myrefimg,frame);
  
    imshow( "Frame", frame );

   
    char c=(char)waitKey(25);
    if(c==27)
      break;
  }

   
  cap.release();

 
  destroyAllWindows();

  return 0;
}
int mymain(Mat imReference,Mat im )
{
 
  Mat imReg, h;

 
  cout << "Aligning images ..." << endl;
  alignImages(im, imReference, imReg, h);

 
  string outFilename("baimg.jpg");
  cout << "Aligned image : " << outFilename << endl;
  imwrite(outFilename, imReg);

 
  cout << "Estimated homography : \n" << h << endl;

  return 0;
}
void alignImages(Mat &im1, Mat &im2, Mat &im1Reg, Mat &h)
{
 
  Mat im1Gray, im2Gray;
  cvtColor(im1, im1Gray, cv::COLOR_BGR2GRAY);
  cvtColor(im2, im2Gray, cv::COLOR_BGR2GRAY);

 
  std::vector<KeyPoint> keypoints1, keypoints2;
  Mat descriptors1, descriptors2;

 
  Ptr<Feature2D> orb = ORB::create(MAX_FEATURES);
  orb->detectAndCompute(im1Gray, Mat(), keypoints1, descriptors1);
  orb->detectAndCompute(im2Gray, Mat(), keypoints2, descriptors2);

 
  std::vector<DMatch> matches;
  Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
  matcher->match(descriptors1, descriptors2, matches, Mat());

 
  std::sort(matches.begin(), matches.end());

 
  const int numGoodMatches = matches.size() * GOOD_MATCH_PERCENT;
  matches.erase(matches.begin()+numGoodMatches, matches.end());

 
  Mat imMatches;
  drawMatches(im1, keypoints1, im2, keypoints2, matches, imMatches);
  imwrite("bcompare.jpg", imMatches);

 
  std::vector<Point2f> points1, points2;

  std::cout << "matches size : " << matches.size() << std::endl;

  for( size_t i = 0; i < matches.size(); i++ )
  {
    points1.push_back( keypoints1[ matches[i].queryIdx ].pt );
    points2.push_back( keypoints2[ matches[i].trainIdx ].pt );
  }

 
  h = findHomography( points1, points2, RANSAC );

 
  warpPerspective(im1, im1Reg, h, im2.size());
}




