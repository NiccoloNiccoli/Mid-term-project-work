#ifdef _OPENMP
    #include <omp.h>
#endif
#include <cstdio>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include<opencv2/imgproc/imgproc.hpp>
#include <chrono>

#define IMAGE_WIDTH 512
#define IMAGE_HEIGHT IMAGE_WIDTH
#define N_CIRCLES 100000

//TODO repeat the experiment with different amounts of circles
struct Circle{
    cv::Point center;
    int radius;
    int color[3]; //bgr
    Circle(cv::Point center, int radius, int blue, int green, int red){this->center = center, this->radius = radius, color[0] = blue, color[1] = green, color[2] = red;}
    Circle(){};
};

void generateCirclesSequential();
void generateCirclesParallel();
int main()
{
#ifdef _OPENMP
    printf("_OPENMP defined\nNum processors: %d\n", omp_get_num_procs());

#endif
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
    //other version
    auto begin = std::chrono::high_resolution_clock::now();
    generateCirclesSequential();
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
    printf("Time measured: %.3f seconds.\n", elapsed.count() * 1e-9);
    //other version 2
    begin = std::chrono::high_resolution_clock::now();
    generateCirclesParallel();
    end = std::chrono::high_resolution_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
    printf("Time measured: %.3f seconds.\n", elapsed.count() * 1e-9);
    cv::waitKey(0);
    return 0;
}
void generateCirclesSequential(){
    cv::Mat bgrchannels[3] = {
            cv::Mat(IMAGE_WIDTH, IMAGE_WIDTH, CV_8UC1, cv::Scalar(255)),
            cv::Mat(IMAGE_WIDTH, IMAGE_WIDTH, CV_8UC1, cv::Scalar(255)),
            cv::Mat(IMAGE_WIDTH, IMAGE_WIDTH, CV_8UC1, cv::Scalar(255))
    };
    cv::Mat backgrounds[3] = {
            cv::Mat(IMAGE_WIDTH, IMAGE_WIDTH, CV_8UC1, cv::Scalar(255)),
            cv::Mat(IMAGE_WIDTH, IMAGE_WIDTH, CV_8UC1, cv::Scalar(255)),
            cv::Mat(IMAGE_WIDTH, IMAGE_WIDTH, CV_8UC1, cv::Scalar(255))
    };
    //define where the circles are
    srand(0);
    Circle circles[N_CIRCLES];
    for (int i = 0; i < N_CIRCLES; i++) {
        int radius = rand() % 70 + 5;
        int colors[3] = {rand() % 256, rand() % 256, rand() % 256};
        circles[i] = Circle{cv::Point(rand() % (IMAGE_WIDTH + 2 * radius) - radius, rand() % (IMAGE_HEIGHT + 2 * radius) - radius), radius, colors[0], colors[1], colors[2]};
    }
    //draw the circles
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < N_CIRCLES; j++) {
            bgrchannels[i].copyTo(backgrounds[i]);
            cv::circle(bgrchannels[i], circles[j].center, circles[j].radius, circles[j].color[i], -1);
            cv::addWeighted(bgrchannels[i], 0.3, backgrounds[i], 1.0 - 0.3, 0.0, bgrchannels[i]);
        }
    }
    cv::Mat image;
    cv::merge(bgrchannels, 3, image);
    cv::imshow("OutputSeq", image);
}

void generateCirclesParallel() {
    cv::Mat bgrchannels[3] = {
            cv::Mat(IMAGE_WIDTH, IMAGE_WIDTH, CV_8UC1, cv::Scalar(255)),
            cv::Mat(IMAGE_WIDTH, IMAGE_WIDTH, CV_8UC1, cv::Scalar(255)),
            cv::Mat(IMAGE_WIDTH, IMAGE_WIDTH, CV_8UC1, cv::Scalar(255))
    };
    cv::Mat backgrounds[3] = {
            cv::Mat(IMAGE_WIDTH, IMAGE_WIDTH, CV_8UC1, cv::Scalar(255)),
            cv::Mat(IMAGE_WIDTH, IMAGE_WIDTH, CV_8UC1, cv::Scalar(255)),
            cv::Mat(IMAGE_WIDTH, IMAGE_WIDTH, CV_8UC1, cv::Scalar(255))
    };
    //define where the circles are
    srand(0);
    Circle circles[N_CIRCLES];
    for (int i = 0; i < N_CIRCLES; i++) {
        int radius = rand() % 70 + 5;
        int colors[3] = {rand() % 256, rand() % 256, rand() % 256};
        circles[i] = Circle{cv::Point(rand() % (IMAGE_WIDTH + 2 * radius) - radius, rand() % (IMAGE_HEIGHT + 2 * radius) - radius), radius, colors[0], colors[1], colors[2]};
    }
    //draw the circles
#pragma omp parallel for default (none) shared (bgrchannels, backgrounds, circles)
    for (int i = 0; i < 3; i++) {
    for (int j = 0; j < N_CIRCLES; j++) {
        bgrchannels[i].copyTo(backgrounds[i]);
        cv::circle(bgrchannels[i], circles[j].center, circles[j].radius, circles[j].color[i], -1);
        cv::addWeighted(bgrchannels[i], 0.3, backgrounds[i], 1.0 - 0.3, 0.0, bgrchannels[i]);
    }
}
    cv::Mat image;
    cv::merge(bgrchannels, 3, image);
    cv::imshow("OutputPar", image);
}





