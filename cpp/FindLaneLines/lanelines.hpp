#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

class LaneLine {
	private:
		cv::Mat final_img;
	public:
		cv::Mat ImgProcessing(cv::Mat const& img);
		cv::Mat Mask(cv::Mat const& img);
		cv::Mat Canny(cv::Mat img);
		cv::Mat DrawLines(cv::Mat const& img, cv::Mat dst, std::vector<float> const& averages);
};
float ewma(float previous, float current, float factor);
