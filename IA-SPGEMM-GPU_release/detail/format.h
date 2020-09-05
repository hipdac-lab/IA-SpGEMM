#ifndef FORMAT_H
#define FORMAT_H

#include <stdio.h>
#include <cusp/multiply.h>
#include <cusp/array2d.h>
#include <cusp/print.h>

typedef cusp::csr_matrix<int,VALUE_TYPE,cusp::host_memory> CUSP_CSR_Host;
typedef cusp::coo_matrix<int,VALUE_TYPE,cusp::device_memory> CUSP_COO_Device;

typedef struct
{
    bool choice=true;
    int row;
    int col;

    VALUE_TYPE* values;
}DenseMatrix;

typedef struct
{
    bool choice=true;
    int row;
    int col;
    int nnz;
    int* row_offset;

    int* row_ind;
    int* col_ind;
    VALUE_TYPE* values;
}CooMatrix;

typedef struct
{
    bool choice=true;
    int row;
    int col;
    int nnz;
    int* row_offset_dev;

    int* row_ind_dev;
    int* col_ind_dev;
    VALUE_TYPE* values_dev;
}CooMatrixDev;

typedef struct
{
    bool choice=true;
    int row;
    int col;
    int nnz;

    int* row_ind;
    int* col_ind;
    VALUE_TYPE* values;
}CsrMatrix;

typedef struct
{
    bool choice=true;
    int row;
    int col;
    int nnz;

    int* row_ind_dev;
    int* col_ind_dev;
    VALUE_TYPE* values_dev;
}CsrMatrixDev;

typedef struct
{
    bool choice=true;
    int row;
    int col;
    int num_diagonals;

    int* diagonal_ind;
    int* diagonal_offsets;
    double* values;
}DiaMatrix;

typedef struct
{
    bool choice=true;
    int row;
    int col;
    int num_diagonals;

    int* diagonal_ind_dev;
    int* diagonal_offsets_dev;
    double* values_dev;
}DiaMatrixDev;


typedef struct
{
    bool choice=true;
    int row;
    int col;
    int nnz;
    int max_nnz_per_row;

    int* nnz_row;
    int* col_ind;
    double* values;
}EllMatrix;

typedef struct
{
    bool choice=true;
    int row;
    int col;
    int nnz;
    int max_nnz_per_row;

    int* nnz_row_dev;
    int* col_ind_dev;
    double* values_dev;
}EllMatrixDev;

typedef struct
{
    bool choice=true;
    int row;
    int col;
    int nnz;

    int* row_ind_dev;
    int* col_ind_dev;
    VALUE_TYPE* values_dev;
}cuSparseMatrix;



#endif // FORMAT_H
