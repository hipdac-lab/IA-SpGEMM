#ifndef ELL_COMMON_H
#define ELL_COMMON_H

#include <stdint.h>

#include <omp.h>
#include <cstring>
#include "immintrin.h"
#include <signal.h>

#include "../common.h"
#include "../utils.h"
#include "../format.h"
#include "../utime.h"

#include "mkl.h"

#define ANONYMOUSLIB_X86_CACHELINE   64


double sizeofell(EllMatrix *A_ell)
{
    double ell_size=0;
    ell_size+=(sizeof(int))*(A_ell->row+A_ell->row*A_ell->max_nnz_per_row+4);
    ell_size+=(sizeof(VALUE_TYPE))*(A_ell->row*A_ell->max_nnz_per_row);
    return ell_size;
}


void CSRtoELL(CsrMatrix *A_csr, EllMatrix *A_ell)
{

    int max_nnz_per_row=0;

    for(int i=0;i<A_csr->row;i++)
    {
        int nnz_this_row=A_csr->row_ind[i+1]-A_csr->row_ind[i];
        max_nnz_per_row=(max_nnz_per_row>nnz_this_row)?max_nnz_per_row:nnz_this_row;
    }

    A_ell->row=A_csr->row;
    A_ell->col=A_csr->col;
    A_ell->nnz=A_csr->nnz;
    A_ell->max_nnz_per_row=max_nnz_per_row;

    A_ell->choice=true;
    if( sizeofell(A_ell)<(50*sizeofcsr(A_csr)))
    {

    A_ell->nnz_row=(int*)malloc1d(A_ell->row,sizeof(int));
    A_ell->col_ind=(int**)malloc2d(A_ell->row,A_ell->max_nnz_per_row,sizeof(int));
    A_ell->values=(VALUE_TYPE**)malloc2d(A_ell->row,A_ell->max_nnz_per_row,sizeof(VALUE_TYPE));

//    for(int i=0;i<A_ell->row;i++)
//        for(int j=0;j<A_ell->max_nnz_per_row;j++)
//            A_ell->col_ind[i][j]=-1;

    for(int i=0;i<A_csr->row;i++)
    {
        int temp_col=0;
        for(int j=A_csr->row_ind[i];j<A_csr->row_ind[i+1];j++)
        {
            A_ell->col_ind[i][temp_col]=A_csr->col_ind[j];
            A_ell->values[i][temp_col]=A_csr->values[j];
            temp_col++;
        }
        A_ell->nnz_row[i]=temp_col;
    }
    }
    else
    {
        A_ell->choice=false;
    }



}


void ELL_MUL_ELL(EllMatrix* A, EllMatrix* B, EllMatrix* C)
{

    C->row=A->row;
    C->col=B->col;
    C->nnz=0;

    C->nnz_row=(int*)malloc1d(C->row,sizeof(int));
    int nnz=0;
    #pragma omp parallel
    {
    int* mask=(int *)malloc((C->col)*sizeof(int));
    for(int i=0;i<C->col;i++) mask[i]=-1;

    #pragma omp for
    for(int i = 0; i < C->row; i++)
    {
        int num_nonzeros = 0;
        for(int jj = 0; jj < A->nnz_row[i]; jj++)
        {
            int j = A->col_ind[i][jj];
            for(int kk = 0; kk < B->nnz_row[j]; kk++)
            {
                int k = B->col_ind[j][kk];
                if(mask[k] != i)
                    {
                        mask[k] = i;
                        num_nonzeros++;
                    } 
            }

        }
		C->nnz_row[i] = num_nonzeros;
    }

    }

    int max_nnz_per_row=0; 
    for(int i = 0; i < C->row; i++)
    {
        max_nnz_per_row=(max_nnz_per_row>C->nnz_row[i])?max_nnz_per_row:C->nnz_row[i];
        nnz+=C->nnz_row[i];
    }

    C->nnz=nnz;
    C->max_nnz_per_row=max_nnz_per_row;

    C->col_ind=(int**)malloc2d(C->row,C->max_nnz_per_row,sizeof(int));
    C->values=(VALUE_TYPE**)malloc2d(C->row,C->max_nnz_per_row,sizeof(VALUE_TYPE));


    int unseen = -1;
    int init = -100;

    #pragma omp parallel
    {
    int* next=(int *)malloc((C->col)*sizeof(int));
    for(int i=0;i<C->col;i++) next[i]=unseen;
    VALUE_TYPE* sums=(VALUE_TYPE *)malloc((C->col)*sizeof(VALUE_TYPE));
    for(int i=0;i<C->col;i++) sums[i]=0.0;

    #pragma omp for
	for (int i = 0; i < C->row; i++)
        {
            int head   = init;
            int length = 0;

            int jj_end = A->nnz_row[i];

            for (int jj = 0; jj < jj_end; jj++)
            {
                int j = A->col_ind[i][jj];
                VALUE_TYPE v = A->values[i][jj];

                int kk_end  = B->nnz_row[j];

                for (int kk = 0; kk < kk_end; kk++)
                {
                    int k = B->col_ind[j][kk];
                    VALUE_TYPE b = B->values[j][kk];

                    sums[k] = sums[k] + v * b ;

                    if (next[k] == unseen)
                    {
                        next[k] = head;
                        head = k;
                        length++;
                    }
                }
            }

            //int offset = C->row_ind[i];

            for (int jj = 0; jj < length; jj++)
            {
                C->col_ind[i][jj] = head;
                C->values[i][jj] = sums[head];

                int temp = head;
                head = next[head];

                // clear arrays
                next[temp] = unseen;
                sums[temp] = 0.0;
            }
        } // end for loop
    } //omp parallel

}



void print_ell(EllMatrix *A_ell)
{

    printf("row:%d col:%d max_nnz_per_row:%d nnz:%d\n",A_ell->row,A_ell->col,A_ell->max_nnz_per_row,A_ell->nnz);

    printf("nnz_row:\n");
    for(int i=0;i<(A_ell->row);i++)
        printf("%d,",A_ell->nnz_row[i]);
    printf("\n");

    printf("col_ind:\n");
    for(int i=0;i<(A_ell->row);i++)
    {
    for(int j=0;j<(A_ell->max_nnz_per_row);j++)
        printf("%d,",A_ell->col_ind[i][j]);
    printf("\n");
    }

    printf("values:\n");
    for(int i=0;i<(A_ell->row);i++)
    {
    for(int j=0;j<(A_ell->max_nnz_per_row);j++)
        printf("%5.2lf,",A_ell->values[i][j]);
    printf("\n");
    }

}


void GetInfo3(EllMatrix *A, double* features)
{

    double param1 = (A->nnz+0.0)/(A->row*A->max_nnz_per_row);
    //printf("%lf,",param1);
    features[0]=param1;

}


void FreeEllMatrix(EllMatrix *A)
{
    free2d(A->values);
    free2d(A->col_ind);
    A->row=0;
    A->col=0;
    A->nnz=0;
    A->max_nnz_per_row=0;
    A->values=NULL;
    A->col_ind=NULL;
    A=NULL;
    //free(A);
}



#endif // DIA_COMMON_H

