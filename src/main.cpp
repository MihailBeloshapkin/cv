#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;

// 1. Converts image to grayscale.
void cvtGray(const Mat& image)
{
	Mat gray;
	cv::cvtColor(image, gray, COLOR_BGR2GRAY);
        imwrite("results/grayScale.png", gray);
}

// 2. Converts image to hue saturation value colorspace.
void cvtHsv(const Mat& image)
{
	Mat hsvImage;
	cv::cvtColor(image, hsvImage, COLOR_BGR2HSV);
        imwrite("results/hsv.png", hsvImage);
}

// 3. Change brightness.
void changeBrightness(const Mat& image, double alpha, int beta)
{
	Mat newImage = Mat::zeros(image.size(), image.type());
	
	for (int y = 0; y < image.rows; y++)
	{
		for (int x = 0; x < image.cols; x++)
		{
			for (int i = 0; i < image.channels(); i++)
		        	newImage.at<Vec3b>(y, x)[i] = saturate_cast<uchar>(alpha * image.at<Vec3b>(y,x)[i] + beta);
		}
	}
	
        imwrite("results/newBrightness.png", newImage);
}

// 4.
void expandByDownBoundary(const Mat& image)
{
	Mat newImage = Mat::zeros(image.size(), image.type());

	for (int y = 0; y < image.rows; y++)
	{
		for (int x = 0; x < image.cols; x++)
		{
			newImage.at<Vec3b>(y, x) = image.at<Vec3b>(image.rows - y, x);	
		}
	}

        imwrite("results/expandedByDown.png", newImage);
}

// 5.
void expandByRightBoundary(const Mat& image)
{
	Mat newImage = Mat::zeros(image.size(), image.type());

	for (int y = 0; y < image.rows; y++)
	{
		for (int x = 0; x < image.cols; x++)
		{
			newImage.at<Vec3b>(y, x) = image.at<Vec3b>(y, image.cols - x);
		}
	}

        imwrite("results/expandedByRight.png", newImage);
}

// 6. Applies gaussian blur to the image.
void applyBlur(const Mat& image)
{
	Mat newImage;
	GaussianBlur(image, newImage, Size(5, 5), 0);
        imwrite("results/Blur.png", newImage);
}

// 7. Apply Canny.
void applyCanny(const Mat& image)
{
	Mat newImage;
        cv::cvtColor(image, newImage, COLOR_BGR2GRAY);
        GaussianBlur(newImage, newImage, Size(5, 5), 0);
        Canny(newImage, newImage, 1, 3, 3);
        imwrite("results/Canny.png", newImage);
}

// 8. Offset right by 10 pixels
void moveRight(const Mat& image)
{
	Mat newImage = Mat::zeros(image.size(), image.type());

	for (int y = 0; y < image.rows; y++)
	{
		for (int x = 0; x < image.cols - 10; x++)
		{
			newImage.at<Vec3b>(y, x + 10) = image.at<Vec3b>(y, x);
		
		}
	}

        imwrite("results/Move.png", newImage);
}

// 9. Rotates image (45 degrees).
void rotateImage(const Mat& image)
{
	double angle = 45;
	Mat newImage;
	Point2f center(image.cols / 2.0, image.rows / 2.0);
	Mat rot = getRotationMatrix2D(center, angle, 1.0);
	cv::Rect2f bbox = cv::RotatedRect(cv::Point2f(), image.size(), angle).boundingRect2f();
       
        rot.at<double>(0,2) += bbox.width / 2.0 - image.cols / 2.0;
        rot.at<double>(1,2) += bbox.height / 2.0 - image.rows / 2.0;

        Mat dst;
        warpAffine(image, dst, rot, bbox.size());
        imwrite("results/Rotation.png", dst);
}

// Image Binarization.
void binarizeImage(const Mat& image)
{
	Mat gray;
	Mat binary;
	cv::cvtColor(image, gray, COLOR_BGR2GRAY);
	cv::threshold(gray, binary, 100, 255, THRESH_BINARY);
        imwrite("results/Binarize.png", binary);
}

// Change contrast of the image.
void changeContrast(const Mat& image)
{
	Mat newImage;
	image.convertTo(newImage, -1, 3, 0);
	imwrite("results/ChangeContrast.png", newImage);
}

void gammaTransform(const Mat& image, float gamma)
{
	Mat dst;
	image.convertTo(dst, CV_64F);
	cv::pow(dst, gamma, dst);
	Mat result;
	dst.convertTo(result, CV_8U);
	imwrite("results/GammaTransform.png", result);
}

int main()
{
	Mat image = imread("results/source.jpg", 1);
	if (!image.data )
	{
		printf("No image");
	}
	
	cvtGray(image);
	cvtHsv(image);
	changeBrightness(image, 1.8, 1);
	changeContrast(image);
	expandByRightBoundary(image);
	expandByDownBoundary(image);
	applyBlur(image);
	applyCanny(image);
	moveRight(image);
	rotateImage(image);
	binarizeImage(image);
	gammaTransform(image, 1.1);
	printf("Changes have already been written to folder /results\n");
	return 0;
}
