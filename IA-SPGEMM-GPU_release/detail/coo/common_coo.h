#ifndef COO_COMMON_H
#define COO_COMMON_H

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


double sizeofcoo(CooMatrix *A_coo)
{
    double coo_size=0;
    coo_size+=(sizeof(int))*(A_coo->row+1+A_coo->nnz+3);
    coo_size+=(sizeof(double))*(A_coo->nnz);
    return coo_size;
}


void CSRtoCOO(CsrMatrix* A_csr, CooMatrix *A_coo)
{

    A_coo->row=A_csr->row;
    A_coo->col=A_csr->col;
    A_coo->nnz=A_csr->nnz;


    if( sizeofcoo(A_coo)<(20*sizeofcsr(A_csr)) )
    {
    A_coo->row_offset=(int*)malloc1d(A_csr->row+1,sizeof(int));
    A_coo->row_ind=(int*)malloc1d(A_csr->nnz,sizeof(int));
    A_coo->col_ind=(int*)malloc1d(A_csr->nnz,sizeof(int));
    A_coo->values=(double*)malloc1d(A_csr->nnz,sizeof(double));

    int index=0;
    for(int i=0;i<A_csr->row;i++)
    {
        A_coo->row_offset[i]=index;
        for(int j=A_csr->row_ind[i];j<A_csr->row_ind[i+1];j++)
        {
            A_coo->row_ind[index]=i;
            A_coo->col_ind[index]=A_csr->col_ind[j];
            A_coo->values[index]=A_csr->values[j];
            index++;
        }
    }
    A_coo->row_offset[A_csr->row]=index;
    }
    else
    {
        A_coo->choice=false;
    }
}





void COO_MUL_COO(CooMatrix* A, CooMatrix* B, CooMatrix* C)
{

    C->row=A->row;
    C->col=B->col;
    C->nnz=0;
    C->row_offset=(int*)malloc1d(C->row+1,sizeof(int));
    int* row_ind=(int *)malloc1d(C->row,sizeof(int));

    #pragma omp parallel
    {
    int* mask=(int *)malloc((C->col)*sizeof(int));
    for(int i=0;i<C->col;i++) mask[i]=-1;

    #pragma omp for
    for(int k=0;k<A->row;k++)
    {
        int nnz_row=0;
        for(int i=A->row_offset[k];i<A->row_offset[k+1];i++)
        {
            int A_i=A->row_ind[i];
            int A_j=A->col_ind[i];
            for(int j=B->row_offset[A_j];j<B->row_offset[A_j+1];j++)
            {
                int B_i=B->row_ind[j];
                int B_j=B->col_ind[j];
                if(mask[B_j] != k)
                {
                    mask[B_j] = k;
                    nnz_row++;
                }
            }
        }
        //printf("nnz : %d\n",nnz_row);
        row_ind[k]=nnz_row;
    }
    }

    C->row_offset[0]=0;
    for(int i=0;i<C->row;i++)
    {
        C->nnz+=row_ind[i];
        C->row_offset[i+1]=C->nnz;
    }
    //printf("nnz : %d\n",C->nnz);

    C->row_ind=(int*)malloc1d(C->nnz,sizeof(int));
    C->col_ind=(int*)malloc1d(C->nnz,sizeof(int));
    C->values=(double*)malloc1d(C->nnz,sizeof(double));
    for(int i=0;i<C->nnz;i++) C->row_ind[i]=-1;
    
    #pragma omp for
    for(int k=0;k<A->row;k++)
    {
        for(int i=A->row_offset[k];i<A->row_offset[k+1];i++)
        {
            int A_i=A->row_ind[i];
            int A_j=A->col_ind[i];
            for(int j=B->row_offset[A_j];j<B->row_offset[A_j+1];j++)
            {
                int B_i=B->row_ind[j];
                int B_j=B->col_ind[j];
                //printf("write : %d,%d\n",A_i,B_j);
                for(int pos=C->row_offset[A_i];pos<C->row_offset[A_i+1];pos++)
                {
                    if(C->row_ind[pos]==-1)
                    {
                        //printf("new : %d,%d,%d\n",A_i,B_j,pos);
                        C->row_ind[pos]=A_i;
                        C->col_ind[pos]=B_j;
                        C->values[pos]=A->values[i]*B->values[j];
                        break;
                    }
                    else if((C->row_ind[pos]==A_i)&&(C->col_ind[pos]==B_j))
                    {
                        //printf("renew : %d,%d,%d\n",A_i,B_j,pos);
                        C->values[pos]+=A->values[i]*B->values[j];
                        break;
                    }
                    else
                    {
                        //continue;
                    }
                }
            }
        }
    }



}




void print_coo(CooMatrix *A_coo)
{
    printf("row:%d col:%d nnz:%d\n",A_coo->row,A_coo->col,A_coo->nnz);

    for(int i=0;i<A_coo->row+1;i++)
        printf("%d,",A_coo->row_offset[i]);
    printf("\n");

    for(int i=0;i<A_coo->nnz;i++)
    {
        printf("%d,%d,%5.2lf\n",A_coo->row_ind[i],A_coo->col_ind[i],A_coo->values[i]);
    }
    printf("\n");

}




void FreeCooMatrix(CooMatrix *A_coo)
{

    if(A_coo->choice)
    {
    free(A_coo->row_ind);
    free(A_coo->col_ind);
    free(A_coo->values);
    }
    A_coo->row=0;
    A_coo->col=0;
    A_coo->nnz=0;
    //free(A_coo);
    A_coo=NULL;
}



#endif // COO_COMMON_H
