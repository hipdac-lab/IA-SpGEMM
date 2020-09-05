#ifndef DENSE_COMMON_H
#define DENSE_COMMON_H

#include <stdint.h>

#include <omp.h>
#include <cstring>
#include "immintrin.h"

#include "../common.h"
#include "../utils.h"
#include "../format.h"
#include "../utime.h"

#include "mkl.h"

#define ANONYMOUSLIB_X86_CACHELINE   64


void CSRtoDENSE(CsrMatrix *A_csr, DenseMatrix *B_dense)
{
    int i,j;
    B_dense->row=A_csr->row;
    B_dense->col=A_csr->col;

    B_dense->values=(double**)malloc2d(B_dense->row,B_dense->col,sizeof(double));

    for(i=0;i<B_dense->row;i++)
        for(j=0;j<B_dense->col;j++)
            B_dense->values[i][j]=0.0;

    for(i=0;i<A_csr->row;i++)
    {
        for(j=A_csr->row_ind[i];j<A_csr->row_ind[i+1];j++)
        {
            B_dense->values[i][A_csr->col_ind[j]]=A_csr->values[j];
        }

    }
}



void print_dense(DenseMatrix *A_dense)
{
    printf("row:%d col:%d\n",A_dense->row,A_dense->col);
    for(int i=0;i<A_dense->row;i++)
    {
        for(int j=0;j<A_dense->col;j++)
        {
            printf("%5.2lf,",A_dense->values[i][j]);
        }
        printf("\n");
    }


}



void FreeDenseMatrix(DenseMatrix *A)
{
    free2d(A->values);
    A->row=0;
    A->col=0;
    A->values=NULL;
    A=NULL;
    //free(A);
}



#endif // DENSE_COMMON_H
