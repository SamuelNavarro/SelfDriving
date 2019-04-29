#include "lanelines.hpp"
#include <opencv2/highgui.hpp>
#include <vector>
#include <numeric>
#include <iostream>


std::vector<float> right_slopes, left_slopes, right_b_x, right_b_y, left_b_x, left_b_y;


int main(int argc, char** argv){

	cv::Vec4f right_line, left_line;
	float right_m, left_m, y, x, slope;;
	cv::Mat img, blur_gray, canny_output, mask_img, dst, final_img;
	std::vector<cv::Vec4i> lines, right_lines, left_lines;;
	std::vector<cv::Point> right_points, left_points;


	const char* file = argv[1];

	img = cv::imread(file, 1);
	dst = img.clone();

	if(img.empty()){
		std::cout << "Error when openening the img " << std::endl;
	}
	
	LaneLine ll;
	blur_gray = ll.ImgProcessing(img);
	canny_output = ll.Canny(blur_gray);
	mask_img = ll.Mask(canny_output);

	cv::HoughLinesP(mask_img, lines, 2, CV_PI / 180, 50, 10, 250);

	for( size_t i = 0; i < lines.size(); i++ )
	{
		cv::Vec4i l = lines[i];
		y = l[3] - l[1];
		x = l[0] - l[2];
		slope = y / x;

		if(slope < 0){
			right_lines.push_back(l);
		}
		else if(slope > 0){
			left_lines.push_back(l);
		}
	}


	for( size_t  i = 0; i < right_lines.size(); i++)
	{
		cv::Point right_ini = cv::Point(right_lines[i][0], right_lines[i][1]);
		cv::Point right_fin = cv::Point(right_lines[i][2], right_lines[i][3]);
		right_points.push_back(right_ini);
		right_points.push_back(right_fin);
	}

	for( size_t  i = 0; i < left_lines.size(); i++)
	{
		cv::Point left_ini = cv::Point(left_lines[i][0], left_lines[i][1]);
		cv::Point left_fin = cv::Point(left_lines[i][2], left_lines[i][3]);
		left_points.push_back(left_ini);
		left_points.push_back(left_fin);
	}


	cv::fitLine(right_points, right_line, cv::DIST_L2, 0, 0.01, 0.01);
	right_m = right_line[1] / right_line[0];
	right_slopes.push_back(right_m);
	float right_avg_m = std::accumulate(right_slopes.begin(), right_slopes.end(), 0.0) / right_slopes.size();
	float right_ewma = ewma(right_avg_m, right_m, 0.5);


	right_b_x.push_back(right_line[2]);
	right_b_y.push_back(right_line[3]);
	float right_b_x_avg = std::accumulate(right_b_x.begin(), right_b_x.end(), 0.0) / right_b_x.size();
	float right_b_y_avg = std::accumulate(right_b_y.begin(), right_b_y.end(), 0.0) / right_b_y.size();
	float right_b_x_ewma = ewma(right_b_x_avg, right_line[2], 0.5);
	float right_b_y_ewma = ewma(right_b_y_avg, right_line[3], 0.5);


	cv::fitLine(left_points, left_line, cv::DIST_L2, 0, 0.01, 0.01);
	left_m = left_line[1] / left_line[0];
	left_slopes.push_back(left_m);
	float left_avg_m = std::accumulate(left_slopes.begin(), left_slopes.end(), 0.0) / left_slopes.size();
	float left_ewma = ewma(left_avg_m, left_m, 0.5);



	left_b_x.push_back(left_line[2]);
	left_b_y.push_back(left_line[3]);
	float left_b_x_avg = std::accumulate(left_b_x.begin(), left_b_x.end(), 0.0) / left_b_x.size();
	float left_b_y_avg = std::accumulate(left_b_y.begin(), left_b_y.end(), 0.0) / left_b_y.size();
	float left_b_x_ewma = ewma(left_b_x_avg, left_line[2], 0.5);
	float left_b_y_ewma = ewma(left_b_y_avg, left_line[3], 0.5);
	

	std::vector<float> averages{ right_m, left_m, left_b_x_avg, left_b_y_avg, right_b_x_avg, right_b_y_avg};


	final_img = ll.DrawLines(img, dst, averages);


	cv::namedWindow("Resultado líneas", cv::WINDOW_AUTOSIZE);
	cv::imshow("Resultado líneas", final_img);

	
	cv::waitKey(0);
	return 0;
}

