#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;

Mat src; Mat src2; Mat src_gray; Mat src_gray2;
int epsilon = 20;
int max_epsilon = 200;
RNG rng(12345);

/// Functions/headers
int ed2(const Point& lhs, const Point& rhs)
{
  return (lhs.x - rhs.x)*(lhs.x - rhs.x) + (lhs.y - rhs.y)*(lhs.y - rhs.y);
}

vector<Point> removeFromContour(const vector<Point>& contour, const vector<int>& defectsIdx)
{
  int minDist = INT_MAX;
  int startIdx;
  int endIdx;

  for (int i = 0; i < defectsIdx.size(); ++i)
  {
    for (int j = i + 1; j < defectsIdx.size(); ++j)
    {
      float dist = ed2(contour[defectsIdx[i]], contour[defectsIdx[j]]);
      if (minDist > dist)
      {
        minDist = dist;
        startIdx = defectsIdx[i];
        endIdx = defectsIdx[j];
      }
    }
  }

  // Check if intervals are swapped
  if (startIdx <= endIdx)
  {
      int len1 = endIdx - startIdx;
      int len2 = contour.size() - endIdx + startIdx;
      if (len2 < len1)
      {
          swap(startIdx, endIdx);
      }
  }
  else
  {
      int len1 = startIdx - endIdx;
      int len2 = contour.size() - startIdx + endIdx;
      if (len1 < len2)
      {
          swap(startIdx, endIdx);
      }
  }

  // Remove unwanted points
  vector<Point> out;
  if (startIdx <= endIdx)
  {
      out.insert(out.end(), contour.begin(), contour.begin() + startIdx);
      out.insert(out.end(), contour.begin() + endIdx, contour.end());
  } 
  else
  {
      out.insert(out.end(), contour.begin() + endIdx, contour.begin() + startIdx);
  }

  return out;
}

void edge_callback(int, void* );

/** @function main */
int main( int argc, char** argv )
{
  /// Load source image and convert it to gray
  src = imread( argv[1], 1 );
  src2 = imread( argv[1], 1);

  // Scale down for testing
  resize(src, src, Size(), 0.5, 0.5);
  resize(src2, src2, Size(), 0.5, 0.5);

  /// Convert image to grayscale
  cvtColor( src, src_gray, CV_BGR2GRAY ); // Conversion to grayscale

  // Quick and dirty fix for poor edge detection: increase contrast at object boundary
  threshold(src_gray, src_gray, 210, 255, 0);
  
  // Different options for blurring, but bilateral is best at preserving edges
  bilateralFilter(src_gray, src_gray2, 5, 80, 80);

  /// Create Window
  char* source_window = "Source";
  namedWindow( source_window, CV_WINDOW_AUTOSIZE );
  imshow( source_window, src );

  createTrackbar( " Epsilon:", "Source", &epsilon, max_epsilon, edge_callback );
  edge_callback( 0, 0 );

  waitKey(0);
  return(0);
}

// Function to translate an image by x, y
Mat translateImg(Mat &img, int x, int y){
    Mat trans_mat = (Mat_<double>(2,3) << 1, 0, x, 0, 1, y);
    warpAffine(img,img,trans_mat,img.size());
    return trans_mat;
}

/** @function thresh_callback */
void edge_callback(int, void* )
{
  Mat canny_output, intermed;
  Mat grad_x, grad_y;
  Mat abs_grad_x, abs_grad_y;
  Mat grad;
  vector<vector<Point> > contours, second_contours;
  vector<Vec4i> hierarchy;

  int maxCont = 0;
  double maxArea = 0;
  vector<int> maxVec;

  /// Detect edges using canny, or Scharr derivatives
  Scharr(src_gray2, grad_x, CV_16S, 1, 0);
  convertScaleAbs(grad_x, abs_grad_x);
  Scharr(src_gray2, grad_y, CV_16S, 0, 1);
  convertScaleAbs(grad_y, abs_grad_y);
  addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, canny_output );

  int erosion_size = 4;   
  cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT,
                      cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1), 
                      cv::Point(erosion_size, erosion_size) );

  cv::dilate(canny_output, canny_output, element);
  cv::erode(canny_output, canny_output, element); 

  canny_output.convertTo(canny_output, CV_8U);
  threshold(canny_output, canny_output, 100, 255, 0); // Careful not to crash out
  // threshold(canny_output, canny_output, 100, 255, 1); // Invert the final image

  namedWindow( "Scharr", CV_WINDOW_AUTOSIZE );
  imshow("Scharr", canny_output);

  // findContours( canny_output, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE ); // Replace approxPolyDP?
  findContours( canny_output, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE );

  // Extract the contours so that we can approximate then with simpler polygons
  vector<vector<Point> > simple_contours;
  findContours( canny_output, simple_contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE );

  contours.resize(simple_contours.size());
  for( size_t i = 0; i < simple_contours.size(); i++ )
      approxPolyDP(Mat(simple_contours[i]), contours[i], epsilon/20, true); // adjust the epsilon parameter for more/less detail

  // Isolate the longest continuous contour from simplified polygons
  // for( int i = 0; i< contours.size(); i++ )
  // {
  //   if(contourArea(contours[i]) > maxArea) {
  //     maxCont = i;
  //     maxArea = contourArea(contours[i]);
  //     // maxVec.push_back(maxCont);
  //   }
  // }

  /// Draw contours
  Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3 );
  drawing.setTo(0); //set to a uniform white/black background

  for (int i = 0; i < contours.size(); i++)
  {
    double a = contourArea(contours[i], false);  //  Find the area of contour
    for (int j = 0; j <= i; j++)
    {
      if (hierarchy[j][3] != -1) // means it has parent contour
      {
        if (a>maxArea)
        {
          maxArea = a;
          maxCont = i;
        }
        // drawContours( drawing, contours, i, Scalar(0,0,255), 1, 8, hierarchy, 0, Point() ); //only the largest contour (maxCont)
      }           
    }
  }

  drawContours( drawing, contours, maxCont, Scalar(0,0,255), 1, 8, hierarchy, 0, Point() ); //only the largest contour (maxCont)
  cvtColor(drawing, drawing, CV_BGR2GRAY);

  // Find new contours from simplified edge approximation
  findContours( drawing, second_contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

  // Excise bad points from this edge
  vector<Point> pts = second_contours[0];

  vector<int> hullIdx;
  convexHull(pts, hullIdx, false);

  vector<Vec4i> defects;
  convexityDefects(pts, hullIdx, defects);

  while(true)
  {
    vector<vector<Point> > tmp;
    tmp.push_back(pts);

    vector<int> defectsIdx;
    const vector<Vec4i>& v = defects;

    for (int i = 0; i < defects.size(); i++) {
      
      float depth = float(v[i][3]) / 256.f;
      if (depth > 200) // adjust to filter by new depth
      {
        defectsIdx.push_back(v[i][2]);

        int startidx = v[i][0]; Point ptStart(pts[startidx]);
        int endidx = v[i][1]; Point ptEnd(pts[endidx]);
        int faridx = v[i][2]; Point ptFar(pts[faridx]);
      }
    }

    if (defectsIdx.size() < 2)
    {
      break;
    }

    pts = removeFromContour(pts, defectsIdx);
    convexHull(pts, hullIdx, false);
    convexityDefects(pts, hullIdx, defects);
  }

  Mat drawing2 = Mat::zeros( canny_output.size(), CV_8UC3 );
  drawing2.setTo(0); //set to a uniform white/black background

  vector<vector<Point> > tmp;
  tmp.push_back(pts);
  drawContours(drawing2, tmp, 0, Scalar(0,0,255), 1);

  // Show in a window
  namedWindow( "Result", CV_WINDOW_AUTOSIZE );
  imshow("Result", drawing2);

  namedWindow( "Basis", CV_WINDOW_AUTOSIZE );
  imshow("Basis", drawing);

  // Draw largest contour onto mask and fill with white
  Mat mask = Mat::zeros(src2.size(), CV_8UC1);
  drawContours(mask, tmp, 0, Scalar(255), CV_FILLED);

  // Crop the unmasked region of the original image
  Mat crop(src2.size(), CV_8UC3);
  crop.setTo(Scalar(0,0,0));
  src.copyTo(crop,mask);
  normalize(mask.clone(), mask, 0.0, 255.0, CV_MINMAX, CV_8UC1);

  namedWindow( "Crop", CV_WINDOW_AUTOSIZE );
  imshow( "Crop", crop);

  // for (int s = 0; s < maxVec.size(); s++) {
  //   printf(" * Contour[%d] ", maxVec[s] );
  // }

}