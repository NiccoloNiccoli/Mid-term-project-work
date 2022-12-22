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
#include <fstream>

#define N_REP 1
#define RADIUS 70
#define MIN_RADIUS 5
#define ALPHA 0.5

// Non parallelizziamo la generazione dei cerchi per avere output identici (dipendono dall'ordine in cui vengono chiamati i rand)
struct Circle{
    cv::Point center;
    int radius;
    int color[3]; //bgr
};

double generateCirclesSequential(int nCircles, int imageHeight, int imageWidth);
double generateCirclesParallel(int nCircles, int imageHeight, int imageWidth, int nThread);
void exportOutputs(std::vector<int> nCircles, std::vector<double> seq, std::vector<double> par, int nThread, int imageSize, const std::string& fileName);
void overlayImage(cv::Mat* src, cv::Mat* overlay);
void overlayImage(cv::Mat* src, cv::Mat* overlay, int y0, int y1);

void overlayImage(cv::Mat* src, cv::Mat* overlay) {
    for (int y = 0; y < src->rows; y++) {
        for (int x = 0; x < src->cols; x++) {
            double alphaSrc = (double)(src->data[y * src->step + x * src->channels() + 3]) / 255;
            double alphaOverlay = (double)(overlay->data[y * overlay->step + x * overlay->channels() + 3]) / 255;
            double alpha = alphaOverlay + alphaSrc * (1 - alphaOverlay);
            unsigned char overlayPx;
            unsigned char srcPx;
            unsigned char finalPx;

            for (int c = 0; c < src->channels() - 1; c++) {
               /* overlayPx = overlay->data[y * overlay->step + x * overlay->channels() + c];
                srcPx = src->data[y * src->step + x * src->channels() + c];
                finalPx = overlayPx + srcPx * (1 - alphaOverlay);
                src->data[y * src->step + src->channels() * x + c] = (finalPx < 255) ? (finalPx + 1) : 255;*/
                overlayPx = overlay->data[y * overlay->step + x * overlay->channels() + c];
                srcPx = src->data[y * src->step + x * src->channels() + c];
                src->data[y * src->step + src->channels() * x + c] = (overlayPx*alphaOverlay + srcPx * alphaSrc * (1-alphaOverlay))/alpha;
            }
            src->data[y * src->step + src->channels() * x + 3] = alpha * 255;
        }
    }
}

void overlayImage(cv::Mat* src, cv::Mat* overlay, int y0, int y1){
    for (int y = y0; y < y1; y++) {
        for (int x = 0; x < src->cols; x++) {
            double alphaSrc = (double)(src->data[y * src->step + x * src->channels() + 3]) / 255;
            double alphaOverlay = (double)(overlay->data[y * overlay->step + x * overlay->channels() + 3]) / 255;
            double alpha = alphaOverlay + alphaSrc * (1 - alphaOverlay);
            unsigned char overlayPx;
            unsigned char srcPx;
            unsigned char finalPx;

            for (int c = 0; c < src->channels() - 1; c++) {
                /* overlayPx = overlay->data[y * overlay->step + x * overlay->channels() + c];
                 srcPx = src->data[y * src->step + x * src->channels() + c];
                 finalPx = overlayPx + srcPx * (1 - alphaOverlay);
                 src->data[y * src->step + src->channels() * x + c] = (finalPx < 255) ? (finalPx + 1) : 255;*/
                overlayPx = overlay->data[y * overlay->step + x * overlay->channels() + c];
                srcPx = src->data[y * src->step + x * src->channels() + c];
                src->data[y * src->step + src->channels() * x + c] = (overlayPx*alphaOverlay + srcPx * alphaSrc * (1-alphaOverlay))/alpha;
            }
            src->data[y * src->step + src->channels() * x + 3] = alpha * 255;
        }
    }
}

int main()
{
#ifdef _OPENMP
    printf("_OPENMP defined\nNum processors: %d\n", omp_get_num_procs());
#endif

    std::vector<int> nThreads = {2, 4, 8};
    int imageSizes [] = {256, 512, 1024};
    std::vector<int> nCircles = {200, 1000, 10000, 100000};
    std::vector<double> seq, par;
    double seqt, part;
    omp_set_dynamic(0);

    using sc = std::chrono::system_clock;
    std::time_t t = sc::to_time_t(sc::now());
    char buf[20];
    strftime(buf, 20, "%d_%m_%Y_%H_%M_%S", localtime(&t));
    std::string s(buf);
    std::string fileName = "../output" + s + ".csv";
    std::cout<<fileName<<std::endl;
    for(int nThread : nThreads){
        omp_set_num_threads(nThread);
        for(int imageSize : imageSizes) {
            for (int n: nCircles) {
                seqt = generateCirclesSequential(n, imageSize, imageSize);
                part = generateCirclesParallel(n, imageSize, imageSize, nThread);
                seq.push_back(seqt);
                par.push_back(part);
            }
            exportOutputs(nCircles, seq, par, nThread, imageSize, fileName);
            seq.clear();
            par.clear();
        }
    }
    return 0;
}

double generateCirclesSequential(int nCircles, int imageHeight, int imageWidth) {
    auto begin = std::chrono::high_resolution_clock::now();
    int numProcs = omp_get_num_procs();
    cv::Mat image = cv::Mat(imageHeight, imageWidth, CV_8UC4, cv::Scalar(255, 255, 255, 255));
    cv::Mat white = cv::Mat(imageHeight, imageWidth, CV_8UC4, cv::Scalar(255, 255, 255, 255));
    cv::Mat images[numProcs];
    for (int i = 0; i < numProcs; i++) {
        images[i] = cv::Mat(imageHeight, imageWidth, CV_8UC4, cv::Scalar(255, 255, 255, 0));
    }
    srand(0);
    Circle circles[nCircles];
    for (int i = 0; i < nCircles; i++) {
        int radius = rand() % RADIUS + MIN_RADIUS;
        circles[i] = Circle{cv::Point(rand() % (imageWidth + 2 * radius) - radius, rand() % (imageHeight + 2 * radius) - radius), radius, rand() % 256, rand() % 256, rand() % 256};
    }
    //TODO PER COMODITà CONSIDERIAMO SOLO NUMERO DI CERCHI MULTIPLI DEL NUMERO DI THREAD
    int minNumCirclesPerImg = (int)(nCircles / numProcs);

    for (int i = 0; i < numProcs; i++) {
        cv::Mat background = cv::Mat(imageHeight, imageWidth, CV_8UC4, cv::Scalar(255, 255, 255, 0));
        for (int j = i*minNumCirclesPerImg; j < (i+1)*minNumCirclesPerImg; j++) {
            images[i].copyTo(background);
            cv::circle(images[i], circles[j].center, circles[j].radius, cv::Scalar(circles[j].color[0], circles[j].color[1], circles[j].color[2], 255), -1);
            cv::addWeighted(images[i], ALPHA, background, 1.0 - ALPHA, 0.0, images[i]);
        }
        //cv::imwrite("../outputSeq2_"+std::to_string(i)+".png", images[i]);
    }

    for (int i = 0; i < numProcs; i++) {
        overlayImage(&white, &images[i]);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
    printf("Seq 2 Time measured: %.4f seconds.\n", elapsed.count() * 1e-9);

    //cv::imshow("OutputSeq_2", white);
    //cv::imwrite("../outputSeq2_.png", white);
    cv::imwrite("../output_" + std::to_string(nCircles) + "_" + std::to_string(imageWidth) + "x" + std::to_string(imageHeight) + "_seq.png", white);
    return elapsed.count() * 1e-9;
}

double generateCirclesParallel(int nCircles, int imageHeight, int imageWidth, int nThread) {
    auto begin = std::chrono::high_resolution_clock::now();
    int numProcs = omp_get_num_procs();
    cv::Mat image = cv::Mat(imageHeight, imageWidth, CV_8UC4, cv::Scalar(255, 255, 255, 255));
    cv::Mat white = cv::Mat(imageHeight, imageWidth, CV_8UC4, cv::Scalar(255, 255, 255, 255));
    cv::Mat images[numProcs];
    for (int i = 0; i < numProcs; i++) {
        images[i] = cv::Mat(imageHeight, imageWidth, CV_8UC4, cv::Scalar(255, 255, 255, 0));
    }
    srand(0);
    Circle circles[nCircles];
    for (int i = 0; i < nCircles; i++) {
        int radius = rand() % RADIUS + MIN_RADIUS;
        circles[i] = Circle{cv::Point(rand() % (imageWidth + 2 * radius) - radius, rand() % (imageHeight + 2 * radius) - radius), radius, rand() % 256, rand() % 256, rand() % 256};
    }
    //TODO PER COMODITà CONSIDERIAMO SOLO NUMERO DI CERCHI MULTIPLI DEL NUMERO DI THREAD
    int minNumCirclesPerImg = nCircles / numProcs;

#pragma omp parallel default(none) shared (images, circles, nCircles, numProcs, minNumCirclesPerImg, white, imageHeight, imageWidth)
#pragma omp for
    for (int i = 0; i < numProcs; i++) {
        cv::Mat background = cv::Mat(imageHeight, imageWidth, CV_8UC4, cv::Scalar(255, 255, 255, 0));
        for (int j = i*minNumCirclesPerImg; j < (i+1)*minNumCirclesPerImg; j++) {
            images[i].copyTo(background);
            cv::circle(images[i], circles[j].center, circles[j].radius, cv::Scalar(circles[j].color[0], circles[j].color[1], circles[j].color[2], 255), -1);
            cv::addWeighted(images[i], ALPHA, background, 1.0 - ALPHA, 0.0, images[i]);
        }
        //cv::imwrite("../outputPar2_"+std::to_string(i)+".png", images[i]);
    }
#pragma omp barrier

    int tileHeight = imageHeight / numProcs;

#pragma omp for
    for (int threadIdx = 0; threadIdx < numProcs; threadIdx++) {
        for (int i = 0; i < numProcs; i++) {
            overlayImage(&white, &images[i], tileHeight * threadIdx, tileHeight * (threadIdx + 1));
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
    printf("Par 2 Time measured: %.4f seconds.\n", elapsed.count() * 1e-9);

    cv::imwrite("../output_"+std::to_string(nCircles)+ + "_" + std::to_string(nThread) + "_" + std::to_string(imageWidth) + "x" + std::to_string(imageHeight) +"_par.png", white);
    cv::Mat bgr = cv::Mat(imageHeight, imageWidth, CV_8UC3, cv::Scalar(255, 255, 255));
    cv::cvtColor(white,bgr,cv::COLOR_BGRA2BGR, 3);
    cv::imwrite("../outputPar2_BGR.png", bgr);


    //cv::imwrite("../output_"+std::to_string(nCircles)+"_par2.png", image);
    return elapsed.count() * 1e-9;
}

void exportOutputs(std::vector<int> nCircles, std::vector<double> seq, std::vector<double> par, int nThreads, int imageSize, const std::string& fileName) {
        std::ofstream outputFile;
        outputFile.open(fileName, std::ios::out | std::ios::app);
        if(outputFile.is_open()) {
            outputFile << std::to_string(nThreads) + " threads, " + std::to_string(imageSize) + "x" + std::to_string(imageSize) + " Number of circles; Sequential version; Parallel version\n";
            for (int i = 0; i < nCircles.size(); i++) {
                outputFile << nCircles[i] << ";" << seq[i] << ";" << par[i] <<  "\n";
            }
            outputFile.close();
        }
}





