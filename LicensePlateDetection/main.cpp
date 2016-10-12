#include <opencv2\opencv.hpp>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>
#include <time.h>
using namespace std;

double cpu_time(void)
{

	double value;
	value = (double) clock () / (double) CLOCKS_PER_SEC;
	return value;

}

void findPlate(cv::Mat image, cv::Point2f *plate)
{

	cv::Mat grayImage, contourImage;
	cv::Mat horizontalElement, verticalElement;
	
	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hierarchy;
	
	int horizontal = 30 + (image.cols/500) * 20;

	verticalElement = cv::getStructuringElement(0, cv::Size(1, 9)); //9
	horizontalElement = cv::getStructuringElement(0, cv::Size(horizontal, 1)); //45
	
	cv::Point2f rect_points[4]; 
	cv::Point2f dim;
	int maxArea = 0;
	for(int p = 0; p < 4; p++)	plate[p] = cv::Point2f(0, 0);

	cv::cvtColor(image, grayImage, CV_BGR2GRAY);
	cv::GaussianBlur(grayImage, contourImage, cv::Size(7, 7), 5);
	cv::adaptiveThreshold(contourImage, contourImage, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY, 3, 0);
	cv::erode(contourImage, contourImage, verticalElement);
	cv::morphologyEx(contourImage, contourImage, 3, horizontalElement);
	cv::findContours( contourImage, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0) );
	std::vector<cv::RotatedRect> minRect( contours.size() );

	for( int i = 0; i< contours.size(); i++ )
    {
		minRect[i] = cv::minAreaRect( cv::Mat(contours[i]) );
		minRect[i].points( rect_points );
		dim.x = abs(rect_points[0].x - rect_points[2].x);
		dim.y = abs(rect_points[0].y - rect_points[2].y);
		
		if(dim.x > dim.y * 3 && dim.x > grayImage.cols * 0.1 && dim.y > grayImage.rows * 0.02 
			&& rect_points[0].y > grayImage.rows * 0.2 && rect_points[0].y < grayImage.rows - grayImage.rows * 0.15
			&& rect_points[0].x > grayImage.cols * 0.2 && rect_points[0].x < grayImage.cols - grayImage.cols * 0.15
			&& rect_points[2].y > grayImage.rows * 0.2 && rect_points[2].y < grayImage.rows - grayImage.rows * 0.15
			&& rect_points[2].x > grayImage.cols * 0.2 && rect_points[2].x < grayImage.cols - grayImage.cols * 0.15
			&& dim.x * dim.y < grayImage.rows * grayImage.cols * 0.125) { 
			if(dim.x * dim.y > maxArea) {
				maxArea = dim.x * dim.y;
				for(int p = 0; p < 4; p++)
					plate[p] = rect_points[p];
			}
		}
			
	}
	
}


int main(int argc, char **argv )
{
	
	cv::Mat image = cv::imread(argv[1]);
	cv::Point2f plate[2];

	double begin = cpu_time();
	findPlate(image, plate);
	double end = cpu_time();
	
	std::cout << "License Plate Detection: " << (end - begin) * 1000 << " ms" << std::endl;
	cv::rectangle(image, plate[0], plate[2], CV_RGB(255, 0, 0), 5, 8);
	while(cv::waitKey(33) != 27) cv::imshow("License Plate Detection", image);
	
	return 0;

}