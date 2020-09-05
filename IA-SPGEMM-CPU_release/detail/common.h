#ifndef COMMON_H
#define COMMON_H

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <math.h>

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

void free2d(VALUE_TYPE** a)
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


#endif // COMMON_H
