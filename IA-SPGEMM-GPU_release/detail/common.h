#ifndef COMMON_H
#define COMMON_H

#include "cuda_runtime.h"

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <math.h>


#define GROUPSIZE_32   32
#define GROUPSIZE_128  128
#define GROUPSIZE_256  256
#define GROUPSIZE_512  512
#define GROUPSIZE_1024 1024
#define GROUPSIZE_4096 4096
#define GROUPSIZE_128_128 16384


void* malloc1d(int rows, int size)
{
    int rowSize = rows * size;
    void *a = (void *)malloc(rowSize);
    memset(a,0,rowSize);
    return a;
}

void** malloc2d(int rows, int cols, int size)
{
    int j;
    int rowSize = cols * size;
    int indexSize = rows * sizeof(void *);
    void **a = (void **) malloc(indexSize + rows* rowSize);
    memset(a,0,indexSize + rows* rowSize);
    char *dataStart = (char *) a + indexSize;
    for(j = 0; j < rows; j++){
        a[j] = dataStart + j * rowSize;
    }
    return a;
}

void free2d(double** a)
{
    free((void**)a);
}

void free2d(long long** a)
{
    free((void**)a);
}

void free2d(int** a)
{
    free((void**)a);
}



void DevMalloc(void** src, size_t size)
{
    cudaError_t Alloc_s = cudaMalloc(src, size);
    if(Alloc_s != cudaSuccess) {
        std::cerr<<"Device malloc failed before line "<<__LINE__<<" in "<<__FILE__<<std::endl;
        cudaFree(*src);
        exit(1);
    }   
    Alloc_s = cudaMemset(*src,0, size);
    if(Alloc_s != cudaSuccess) {
        std::cerr<<"Device malloc failed before line "<<__LINE__<<" in "<<__FILE__<<std::endl;
        cudaFree(*src);
        exit(1);
    }   
    return;
}

void DevUpload(void* dst, void* src, size_t size)
{
    cudaError_t Copy_s = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
    if(Copy_s != cudaSuccess) {
        std::cerr<<"Device upload failed before line "<<__LINE__<<" in "<<__FILE__<< " Error is "<<cudaGetErrorString(Copy_s)<<std::endl;
        exit(1);
    }   
    return;
}

void DevDownload(void* dst, void* src, size_t size)
{
    cudaError_t Copy_s = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
    if(Copy_s != cudaSuccess) {
        std::cerr<<"Device download failed before line "<<__LINE__<<" in "<<__FILE__<< " Error is "<<cudaGetErrorString(Copy_s)<<std::endl;
        exit(1);
    }   
    return;
}






#endif // COMMON_H
