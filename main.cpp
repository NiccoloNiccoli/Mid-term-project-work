#include <cstdio>
#include <omp.h>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include<opencv2/imgproc/imgproc.hpp>

#define IMAGE_WIDTH 512

int main()
{
    std::string location="C:\\Users\\super\\OneDrive\\Immagini\\paciocconelmare.jpg";
    std::cout<<location<<std::endl;
    cv::Mat image (IMAGE_WIDTH, IMAGE_WIDTH, CV_8UC3, cv::Scalar(255,255, 255));
    cv::Mat background;

/*    cv::Mat roi = image(cv::Rect(100, 100, 300, 300));
    cv::Mat color(roi.size(), CV_8UC3, cv::Scalar(0, 125, 125));
    double alpha = 0.3;
    cv::addWeighted(color, alpha, roi, 1.0 - alpha , 0.0, roi);
*/
    double alpha = 0.3;
    int radius = 75; //Declaring the radius
    cv::Scalar line_Color(100, 0, 230);//Color of the circle
    int thickness = -1;//thickens of the line
    for(int i=0; i<3; i++){
        image.copyTo(background);
        cv::Point center = cv::Point(100+25*i, 100+25*i);
        cv::circle(image, center, radius, cv::Scalar(i==2?255:0, i==1?255:0, i==0?255:0), thickness);
        cv::Mat roi = image(cv::Rect(center.x - radius, center.y - radius, radius*2, radius*2));
        cv::addWeighted(image, alpha, background, 1.0 - alpha , 0.0, image);
    }

    cv::imshow("Output", image);
    cv::waitKey(0);
    return 0;
}





