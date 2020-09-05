#ifndef FORMAT_H
#define FORMAT_H

#include <stdio.h>
#include "mkl.h"

typedef struct
{
    bool choice=true;
    int row=0;
    int col=0;

    VALUE_TYPE** values=NULL;
}DenseMatrix;

typedef struct
{
    bool choice=true;
    int row=0;
    int col=0;
    int nnz=0;
    int* row_offset=NULL;

    int* row_ind=NULL;
    int* col_ind=NULL;
    VALUE_TYPE* values=NULL;
}CooMatrix;

typedef struct
{
    bool choice=true;
    int row=0;
    int col=0;
    int nnz=0;

    int* row_ind=NULL;
    int* col_ind=NULL;
    VALUE_TYPE* values=NULL;
}CsrMatrix;

typedef struct
{
    bool choice=true;
    int row=0;
    int col=0;
    int nnz=0;

    MKL_INT* row_ind=NULL;
    MKL_INT* col_ind=NULL;
    VALUE_TYPE* values=NULL;
}MKLMatrix;

typedef struct
{
    bool choice=true;
    int row=0;
    int col=0;
    int num_diagonals=0;

    int* diagonal_ind=NULL;
    int* diagonal_offsets=NULL;
    VALUE_TYPE** values=NULL;
}DiaMatrix;

typedef struct
{
    bool choice=true;
    int row=0;
    int col=0;
    int nnz=0;
    int max_nnz_per_row=0;

    int* nnz_row=NULL;
    int** col_ind=NULL;
    VALUE_TYPE** values=NULL;
}EllMatrix;



#endif // FORMAT_H
