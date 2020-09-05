#ifndef UTIME_H
#define UTIME_H

#include <cuda_runtime.h>

struct anonymouslib_timer_gpu {
    cudaEvent_t start1,stop1;
    void create() {
        cudaEventCreate(&start1);  
        cudaEventCreate(&stop1);  
    }


    void start() {
        cudaEventRecord(start1, 0);
    }

    double stop() {
        cudaEventRecord(stop1, 0);
        cudaEventSynchronize(stop1);
        float elapsedTime = 0;
        cudaEventElapsedTime(&elapsedTime, start1, stop1);
        return elapsedTime;
    }
};


struct anonymouslib_timer_cpu {
    timeval t1, t2;
    struct timezone tzone;

    void start() {
        gettimeofday(&t1, &tzone);
    }

    double stop() {
        gettimeofday(&t2, &tzone);
        double elapsedTime = 0;
        elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;      // sec to ms
        elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;   // us to ms
        return elapsedTime;
    }
};


#endif // UTIME_H
