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


#define IMAGE_WIDTH 512
#define IMAGE_HEIGHT IMAGE_WIDTH
#define N_REP 5

// Non parallelizziamo la generazione dei cerchi per avere output identici (dipendono dall'ordine in cui vengono chiamati i rand)
struct Circle{
    cv::Point center;
    int radius;
    int color[3]; //bgr
};

double generateCirclesSequential(int nCircles);
double generateCirclesParallel(int nCircles);
double generateCirclesSequential_2(int Circles);
double generateCirclesParallel_2(int nCircles);
double generateCirclesParallelArray(int n);
void exportOutputs(std::vector<int> nCircles,std::vector<double> seq,std::vector<double> par, std::vector<double> parArray);

void overlayImage(cv::Mat* src, cv::Mat* overlay) {
    for (int y = 0; y < src->rows; y++) {
        for (int x = 0; x < src->cols; x++) {
            double alphaSrc = src->data[y * src->step + x * src->channels() + 3] / 255;
            double alphaOverlay = overlay->data[y * overlay->step + x * overlay->channels() + 3] / 255;
            double alpha = alphaOverlay + alphaSrc * (1 - alphaOverlay);
            unsigned char overlayPx;
            unsigned char srcPx;
            unsigned char finalPx;

            for (int c = 0; c < src->channels() - 1; c++) {
                overlayPx = overlay->data[y * overlay->step + x * overlay->channels() + c];
                srcPx = src->data[y * src->step + x * src->channels() + c];
                finalPx = overlayPx + srcPx * (1 - alphaOverlay);
                src->data[y * src->step + src->channels() * x + c] = (finalPx < 255) ? (finalPx + 1) : 255;
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
    //generateCirclesSequential_2(2400);
    generateCirclesParallel_2(16);
    generateCirclesParallel(16);
    /*
    std::vector<double> seq, par, parArray;
    double seqt = 0, part = 0, parArrayt = 0;
    std::vector<int> nCircles = {10, 100, 1000, 10000};
    for(int n : nCircles) {
        for (int i = 0; i < N_REP; i++) {
            seqt += generateCirclesSequential(n);
            part += generateCirclesParallel(n);
            parArrayt += generateCirclesParallelArray(n);
        }
        seq.push_back(seqt / N_REP);
        par.push_back(part / N_REP);
        parArray.push_back(parArrayt / N_REP);
    }
    exportOutputs(nCircles, seq, par, parArray);
     */
    return 0;
}

double generateCirclesSequential_2(int nCircles){
    auto begin = std::chrono::high_resolution_clock::now();
    int numProcs = omp_get_num_procs();
    cv::Mat image = cv::Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC4, cv::Scalar(255, 255, 255, 255));
    cv::Mat background = cv::Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC4, cv::Scalar(255, 255, 255, 255));
    cv::Mat images[numProcs];
    for (int i = 0; i < numProcs; i++) {
        images[i] = cv::Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC4, cv::Scalar(255, 255, 255, 0));
    }
    srand(0);
    Circle circles[nCircles];
    for (int i = 0; i < nCircles; i++) {
        int radius = rand() % 70 + 5;
        circles[i] = Circle{cv::Point(rand() % (IMAGE_WIDTH + 2 * radius) - radius, rand() % (IMAGE_HEIGHT + 2 * radius) - radius), radius, rand() % 256, rand() % 256, rand() % 256};
    }
    //TODO PER COMODITà CONSIDERIAMO SOLO NUMERO DI CERCHI MULTIPLI DEL NUMERO DI THREAD
    int minNumCirclesPerImg = (int)(nCircles / numProcs);

    for (int i = 0; i < numProcs; i++) {
        for (int j = i*minNumCirclesPerImg; j < (i+1)*minNumCirclesPerImg; j++) {
            cv::circle(images[i], circles[j].center, circles[j].radius, cv::Scalar(circles[j].color[0], circles[j].color[1], circles[j].color[2], 76.5), -1);
            cv::imwrite("../outputSeq2_"+std::to_string(i)+".png", images[i]);
        }
    }

    for (int i = 0; i < numProcs; i++) {
        overlayImage(&background, &images[i]);
    }
    cv::imshow("OutputSeq_2", background);
    cv::imwrite("../outputSeq2_.png", background);


    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
    printf("Seq 2 Time measured: %.4f seconds.\n", elapsed.count() * 1e-9);
    //cv::imwrite("../output_"+std::to_string(nCircles)+"_par2.png", image);
    return elapsed.count() * 1e-9;
}

double generateCirclesParallel_2(int nCircles){
    auto begin = std::chrono::high_resolution_clock::now();
    int numProcs = omp_get_num_procs();
    cv::Mat image = cv::Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC4, cv::Scalar(255, 255, 255, 255));
    cv::Mat white = cv::Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC4, cv::Scalar(255, 255, 255, 255));
    cv::Mat images[numProcs];
    for (int i = 0; i < numProcs; i++) {
        images[i] = cv::Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC4, cv::Scalar(255, 255, 255, 0));
    }
    srand(0);
    Circle circles[nCircles];
    for (int i = 0; i < nCircles; i++) {
        int radius = rand() % 70 + 5;
        circles[i] = Circle{cv::Point(rand() % (IMAGE_WIDTH + 2 * radius) - radius, rand() % (IMAGE_HEIGHT + 2 * radius) - radius), radius, rand() % 256, rand() % 256, rand() % 256};
    }
    //TODO PER COMODITà CONSIDERIAMO SOLO NUMERO DI CERCHI MULTIPLI DEL NUMERO DI THREAD
    int minNumCirclesPerImg = (int)(nCircles / numProcs);

#pragma omp parallel for default(none) shared (images, circles, nCircles, numProcs, minNumCirclesPerImg)
    for (int i = 0; i < numProcs; i++) {
        cv::Mat background = cv::Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC4, cv::Scalar(255, 255, 255, 0));
        for (int j = i*minNumCirclesPerImg; j < (i+1)*minNumCirclesPerImg; j++) {
            images[i].copyTo(background);
            cv::circle(images[i], circles[j].center, circles[j].radius, cv::Scalar(circles[j].color[0], circles[j].color[1], circles[j].color[2], 255), -1);
            cv::addWeighted(images[i], 0.3, background, 1.0 - 0.3, 0.0, images[i]);
        }
        cv::imwrite("../outputPar2_"+std::to_string(i)+".png", images[i]);
    }

    for (int i = 0; i < numProcs; i++) {
        //TODO I COLORI VENGONO LEGGERMENTE DIVERSI (-1 SU 255 A OGNI FUSIONE DI LAYER) PER APPROSSIMAZIONE DEL NUOVO COLORE
        std::cout<<"rosso "<<i<<" "<<static_cast<unsigned>(white.data[510 * white.step + 510 * white.channels() + 2])<<std::endl;
        std::cout<<"verde "<<i<<" "<<static_cast<unsigned>(white.data[510 * white.step + 510 * white.channels() + 1])<<std::endl;
        std::cout<<"blu "<<i<<" "<<static_cast<unsigned>(white.data[510 * white.step + 510 * white.channels()])<<std::endl;
        overlayImage(&white, &images[i]);
    }

    cv::imwrite("../outputPar2_.png", white);
    cv::Mat bgr = cv::Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC3, cv::Scalar(255, 255, 255));
    cv::cvtColor(white,bgr,cv::COLOR_BGRA2BGR, 3);
    cv::imwrite("../outputPar2_BGR.png", bgr);


    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
    printf("Par 2 Time measured: %.4f seconds.\n", elapsed.count() * 1e-9);
    //cv::imwrite("../output_"+std::to_string(nCircles)+"_par2.png", image);
    return elapsed.count() * 1e-9;
}




double generateCirclesSequential(int nCircles){
    auto begin = std::chrono::high_resolution_clock::now();
    cv::Mat bgrchannels[3] = {
            cv::Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC1, cv::Scalar(255)),
            cv::Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC1, cv::Scalar(255)),
            cv::Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC1, cv::Scalar(255))
    };
    cv::Mat backgrounds[3] = {
            cv::Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC1, cv::Scalar(255)),
            cv::Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC1, cv::Scalar(255)),
            cv::Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC1, cv::Scalar(255))
    };
    srand(0);
    Circle circles[nCircles];
    for (int i = 0; i < nCircles; i++) {
        int radius = rand() % 70 + 5;
        circles[i] = Circle{cv::Point(rand() % (IMAGE_WIDTH + 2 * radius) - radius, rand() % (IMAGE_HEIGHT + 2 * radius) - radius), radius, rand() % 256, rand() % 256, rand() % 256};
    }
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < nCircles; j++) {
            bgrchannels[i].copyTo(backgrounds[i]);
            cv::circle(bgrchannels[i], circles[j].center, circles[j].radius, circles[j].color[i], -1);
            cv::addWeighted(bgrchannels[i], 0.3, backgrounds[i], 1.0 - 0.3, 0.0, bgrchannels[i]);
        }
    }
    cv::Mat image;
    cv::merge(bgrchannels, 3, image);
    cv::imshow("OutputSeq", image);
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
    printf("Time measured: %.4f seconds.\n", elapsed.count() * 1e-9);
    cv::imwrite("../output_"+std::to_string(nCircles)+"_seq.png", image);
    return elapsed.count() * 1e-9;
}

double generateCirclesParallel(int nCircles) {
    auto begin = std::chrono::high_resolution_clock::now();
    cv::Mat bgrchannels[3] = {
            cv::Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC1, cv::Scalar(255)),
            cv::Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC1, cv::Scalar(255)),
            cv::Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC1, cv::Scalar(255))
    };
    cv::Mat backgrounds[3] = {
            cv::Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC1, cv::Scalar(255)),
            cv::Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC1, cv::Scalar(255)),
            cv::Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC1, cv::Scalar(255))
    };
    srand(0);
    Circle circles[nCircles];

    for (int i = 0; i < nCircles; i++) {
        int radius = rand() % 70 + 5;
        circles[i] = Circle{cv::Point(rand() % (IMAGE_WIDTH + 2 * radius) - radius, rand() % (IMAGE_HEIGHT + 2 * radius) - radius), radius, rand() % 256, rand() % 256, rand() % 256};
    }

   #pragma omp parallel for default(none) shared (bgrchannels, backgrounds, circles, nCircles)
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < nCircles; j++) {
            bgrchannels[i].copyTo(backgrounds[i]);
            cv::circle(bgrchannels[i], circles[j].center, circles[j].radius, circles[j].color[i], -1);
            cv::addWeighted(bgrchannels[i], 0.3, backgrounds[i], 1.0 - 0.3, 0.0, bgrchannels[i]);
        }
        cv::imwrite("../outputPar_"+std::to_string(i)+".png", bgrchannels[i]);
    }
    cv::Mat image;
    cv::merge(bgrchannels, 3, image);
    cv::imshow("OutputPar", image);
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
    printf("Time measured: %.4f seconds.\n", elapsed.count() * 1e-9);
    cv::imwrite("../output_"+std::to_string(nCircles)+"_par.png", image);
    return elapsed.count() * 1e-9;
}

double generateCirclesParallelArray(int nCircles) {
   auto begin = std::chrono::high_resolution_clock::now();
   cv::Point centers[nCircles];
   int radiuses[nCircles];
   int b[nCircles], g[nCircles], r[nCircles];
    cv::Mat bgrchannels[3] = {
            cv::Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC1, cv::Scalar(255)),
            cv::Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC1, cv::Scalar(255)),
            cv::Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC1, cv::Scalar(255))
    };
    cv::Mat backgrounds[3] = {
            cv::Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC1, cv::Scalar(255)),
            cv::Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC1, cv::Scalar(255)),
            cv::Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC1, cv::Scalar(255))
    };
   srand(0);
   for (int i = 0; i < nCircles; i++) {
       int radius = rand()%70 + 5;
       radiuses[i] = radius;
       centers[i] = cv::Point(rand() % (IMAGE_WIDTH + 2 * radius) - radius, rand() % (IMAGE_HEIGHT + 2 * radius) - radius);
       b[i] = rand()%256;
       g[i] = rand()%256;
       r[i] = rand()%256;
   }
#pragma omp parallel for default(none) shared (bgrchannels, backgrounds, nCircles, centers, radiuses, b, g, r)
   for (int i = 0; i < 3; i++) {
       for (int j = 0; j < nCircles; j++) {
           bgrchannels[i].copyTo(backgrounds[i]);
           cv::circle(bgrchannels[i], centers[j], radiuses[j], i==0?b[j]:(i==1?g[j]:r[j]), -1);
           cv::addWeighted(bgrchannels[i], 0.3, backgrounds[i], 1.0 - 0.3, 0.0, bgrchannels[i]);
       }
   }
   cv::Mat image;
   cv::merge(bgrchannels, 3, image);
   cv::imshow("OutputParArray", image);
   auto end = std::chrono::high_resolution_clock::now();
   auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
   printf("Time measured: %.4f seconds.\n", elapsed.count() * 1e-9);
    cv::imwrite("../output_"+std::to_string(nCircles)+"_parArray.png", image);
    return elapsed.count() * 1e-9;
}

void exportOutputs(std::vector<int> nCircles,std::vector<double> seq,std::vector<double> par, std::vector<double> parArray) {
    using sc = std::chrono::system_clock ;
    std::time_t t = sc::to_time_t(sc::now());
    char buf[20];
    strftime(buf, 20, "%d_%m_%Y_%H_%M_%S", localtime(&t));
    std::string s(buf);
    std::ofstream outputFile;
    std::string fileName = "../output" + s + ".csv";
    std::cout<<fileName<<std::endl;
    outputFile.open(fileName, std::ios::out | std::ios::app);
    if(outputFile.is_open()) {
        std::cout<<"ok"<<std::endl;
        outputFile << "Number of circles; Sequential version; Parallel version; Parallel version with Array\n";
        for (int i = 0; i < nCircles.size(); i++) {
            outputFile << nCircles[i] << ";" << seq[i] << ";" << par[i] << ";" << parArray[i] << "\n";
        }
        outputFile.close();
    }
}





