#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <numeric>
#include "lanelines.hpp"



cv::Mat LaneLine::ImgProcessing(cv::Mat const& img){
	cv::Mat gray_img, output_img;
	cv::cvtColor(img, gray_img, cv::COLOR_RGB2GRAY);
	cv::GaussianBlur(gray_img, output_img, cv::Size(5,5), 0);

	return output_img;
}


cv::Mat LaneLine::Mask(cv::Mat const& img_edges){
	cv::Mat masked_edges;
	cv::Mat mask = cv::Mat::zeros(img_edges.size(), img_edges.type());

	std::vector<cv::Point> point;
	point.push_back(cv::Point(0, img_edges.size().height));
	point.push_back(cv::Point(470, 300));
	point.push_back(cv::Point(490, 300));
	point.push_back(cv::Point(img_edges.size().width + 80 , img_edges.size().height));
	cv::fillConvexPoly(mask, point, cv::Scalar(255, 0, 0));
	cv::bitwise_and(img_edges, mask, masked_edges);

	return masked_edges;
}	


cv::Mat LaneLine::Canny(cv::Mat const img){
	cv::Mat canny_output;
	cv::Canny(img, canny_output, 50, 150);

	return canny_output;
}



cv::Mat LaneLine::DrawLines(cv::Mat const& img, cv::Mat dst, std::vector<float> const& averages){

	float y_ini = img.rows;
	float y_fin = img.rows / 1.6;
	float right_ini_x = ((y_ini - averages[5]) / averages[0]) + averages[4];
	float right_fin_x = ((y_fin - averages[5]) / averages[0]) + averages[4];
	float left_ini_x = ((y_ini - averages[3]) / averages[1]) + averages[2];
	float left_fin_x = ((y_fin - averages[3]) / averages[1]) + averages[2];

	cv::line(dst, cv::Point(right_ini_x, y_ini), cv::Point(right_fin_x, y_fin), cv::Scalar(0,0,255), 9, cv::LINE_AA);
	cv::line(dst, cv::Point(left_ini_x, y_ini), cv::Point(left_fin_x, y_fin), cv::Scalar(0,0,255), 9, cv::LINE_AA);

	cv::addWeighted(img, 0.5, dst, 0.5, 0, final_img);

	return final_img;
}


float ewma(float previous, float current, float factor){
	return (factor * previous) + (1.0 - factor) * current;
}
