#include <cstdio>
#include <omp.h>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include<opencv2/imgproc/imgproc.hpp>
#include <chrono>

#define IMAGE_WIDTH 512
struct Circle{
    int x;
    int y;
    int z;
    int radius;
    int blue;
    int green;
    int red;
};

bool compare( Circle a, Circle b){
    if(a.z > b.z)
        return 1;
    else
        return 0;
}

int main()
{
    // Start measuring time
    auto begin = std::chrono::high_resolution_clock::now();

    cv::Mat image (IMAGE_WIDTH, IMAGE_WIDTH, CV_8UC3, cv::Scalar(255,255, 255));
    cv::Mat background;

    //creating circles
    /*
     * Let’s consider a structure that represents 3D points.
• If we use all the coordinates for a computation (e.g. compute a
distance) AoS makes sense.
• If we need only some coordinates (e.g. compute a gradient along a
specific axis) then SoA is better.
• The optimal data layout depends entirely on usage and the particular
data access patterns.
• In mixed use cases, which are likely to appear in real applications,
sometimes the structure variables are used together and sometimes not.
• Generally, the AoS layout performs better overall on CPUs, while the
SoA layout performs better on GPUs. If there is enough variability test
for a particular usage pattern.
     */
    int n = 10000;
    Circle circles[n];
    srand(0);
    for (int i=0; i<n ; i++) {
        struct Circle c;
        c.radius = rand()%70 + 5;
        c.x = rand()%(image.cols - 2*c.radius) + c.radius;
        c.y = rand()%(image.rows - 2*c.radius) + c.radius;
        c.z = rand()%20 + 1;
        c.blue = rand()%256;
        c.green = rand()%256;
        c.red = rand()%256;
        circles[i] = c;
    }

    std::sort(circles, circles+n, compare);

    double alpha = 0.3;
    int thickness = -1;//thickens of the line

    for(int i=0; i<n; i++){
        image.copyTo(background);
        cv::Point center = cv::Point(circles[i].x, circles[i].y);
        //std::cout<<circles[i].z<<std::endl;
        cv::circle(image, center, circles[i].radius, cv::Scalar(circles[i].blue, circles[i].green, circles[i].red), thickness);
        cv::Mat roi = image(cv::Rect(center.x - circles[i].radius, center.y - circles[i].radius, circles[i].radius*2, circles[i].radius*2));
        cv::addWeighted(image, alpha, background, 1.0 - alpha , 0.0, image);
    }

    cv::imshow("Output", image);
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
    printf("Time measured: %.3f seconds.\n", elapsed.count() * 1e-9);
    cv::waitKey(0);
    return 0;
}





