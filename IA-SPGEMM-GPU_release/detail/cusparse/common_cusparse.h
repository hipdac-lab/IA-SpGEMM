#ifndef CUSPARSE_COMMON_H
#define CUSPARSE_COMMON_H

#include "../common.h"
#include "../utils.h"
#include "../format.h"
#include "../utime.h"

#include "cusparse.h"

void CSR_to_CUSPARSE(CsrMatrix* A_csr, cuSparseMatrix* A_csr_cusparse)
{
 
    A_csr_cusparse->row=A_csr->row;
    A_csr_cusparse->col=A_csr->col;
    A_csr_cusparse->nnz=A_csr->nnz;

    DevMalloc((void**)&A_csr_cusparse->row_ind_dev,sizeof(int)*(A_csr_cusparse->row+1));
    DevMalloc((void**)&A_csr_cusparse->col_ind_dev,sizeof(int)*(A_csr_cusparse->nnz));
    DevMalloc((void**)&A_csr_cusparse->values_dev,sizeof(double)*(A_csr_cusparse->nnz));

    DevUpload(A_csr_cusparse->row_ind_dev, A_csr->row_ind, sizeof(int)*(A_csr_cusparse->row+1));
    DevUpload(A_csr_cusparse->col_ind_dev, A_csr->col_ind, sizeof(int)*(A_csr_cusparse->nnz));
    DevUpload(A_csr_cusparse->values_dev, A_csr->values, sizeof(double)*(A_csr_cusparse->nnz));

}


double CUSPARSE_MUL_CUSPARSE(cuSparseMatrix* A_cusparse, cuSparseMatrix* B_cusparse, cuSparseMatrix* C_cusparse, anonymouslib_timer_gpu* ref_timer_gpu)
{

	cusparseStatus_t status;
    cusparseHandle_t handle = 0;
    cusparseMatDescr_t descrA, descrB, descrC;

	descrA = 0;
    descrB = 0;
    descrC = 0;

    C_cusparse->row=A_cusparse->row;
    C_cusparse->col=B_cusparse->col;


	status = cusparseCreate(&handle);
    if(status != CUSPARSE_STATUS_SUCCESS)
    {
        std::cout << "cuSPARSE Library initialization failed" << std::endl;
    }

    status = cusparseCreateMatDescr(&descrA);
    if(status != CUSPARSE_STATUS_SUCCESS)
    {
        std::cout << "Matrix descriptoe initialization failed!" << std::endl;
    }
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);

    status = cusparseCreateMatDescr(&descrB);
    if(status != CUSPARSE_STATUS_SUCCESS)
    {
        std::cout << "Matrix descriptoe initialization failed!" << std::endl;
    }
    cusparseSetMatType(descrB, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrB, CUSPARSE_INDEX_BASE_ZERO);

    status = cusparseCreateMatDescr(&descrC);
    if(status != CUSPARSE_STATUS_SUCCESS)
    {
        std::cout << "Matrix descriptoe initialization failed!" << std::endl;
    }
    cusparseSetMatType(descrC, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrC, CUSPARSE_INDEX_BASE_ZERO);
	
    ref_timer_gpu->start();

    DevMalloc((void**)&C_cusparse->row_ind_dev,sizeof(int)*(C_cusparse->row+1));

	cusparseXcsrgemmNnz(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, A_cusparse->row, B_cusparse->col, A_cusparse->col,
            descrA, A_cusparse->nnz, A_cusparse->row_ind_dev, A_cusparse->col_ind_dev,
            descrB, B_cusparse->nnz, B_cusparse->row_ind_dev, B_cusparse->col_ind_dev,
            descrC, C_cusparse->row_ind_dev, &C_cusparse->nnz);

    cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);

    DevMalloc((void**)&C_cusparse->col_ind_dev,sizeof(int)*(C_cusparse->nnz));
    DevMalloc((void**)&C_cusparse->values_dev,sizeof(double)*(C_cusparse->nnz));

	cusparseDcsrgemm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, A_cusparse->row, B_cusparse->col, A_cusparse->col,
            descrA, A_cusparse->nnz,A_cusparse->values_dev, A_cusparse->row_ind_dev, A_cusparse->col_ind_dev,
            descrB, B_cusparse->nnz,B_cusparse->values_dev, B_cusparse->row_ind_dev, B_cusparse->col_ind_dev,
            descrC, C_cusparse->values_dev, C_cusparse->row_ind_dev, C_cusparse->col_ind_dev);

    double time_cusparse = ref_timer_gpu->stop();
    return time_cusparse;

}




__global__ void print_cusparse_cu(int row, int col, int nnz, int* row_ind, int* col_ind, double* values)
{

    printf("row:%d col:%d nnz:%d\n",row,col,nnz);
    for(int i=0;i<=row;i++)
    {
        printf("%d,",row_ind[i]);
    }
    printf("\n");

    for(int i=0;i<nnz;i++)
    {
        printf("%d,",col_ind[i]);
    }
    printf("\n");

    for(int i=0;i<nnz;i++)
    {
        printf("%.2lf,",values[i]);
    }
    printf("\n");

}

void print_cusparse(cuSparseMatrix *A)
{

print_cusparse_cu<<<1,1>>>(A->row, A->col, A->nnz, A->row_ind_dev, A->col_ind_dev, A->values_dev);

}

__global__ void getsum_cusparse_cu(double* sum, int nnz, double* values)
{
for(int i=0;i<nnz;i++)
    sum[0]+=values[i];
}

double getsum_cusparse(cuSparseMatrix *A)
{
double sum;
double *sum_dev;
DevMalloc((void**)&sum_dev,sizeof(double));
getsum_cusparse_cu<<<1,1>>>(sum_dev, A->nnz, A->values_dev);
DevDownload(&sum, sum_dev, sizeof(double));
return sum;
}

double sizeofcusparse(cuSparseMatrix *A)
{
    double cusparse_size=0;
    cusparse_size+=(sizeof(int))*(A->row+1+A->nnz+3);
    cusparse_size+=(sizeof(double))*(A->nnz);
    return cusparse_size;
}


void FreecuSparseMatrix(cuSparseMatrix *A)
{
    cudaFree(A->row_ind_dev);
    cudaFree(A->col_ind_dev);
    cudaFree(A->values_dev);
    A->row=0;
    A->col=0;
    A->nnz=0;
    A=NULL;
}


#endif // CUSPARSE_COMMON_H
