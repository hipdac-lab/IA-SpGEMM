#ifndef CSR_COMMON_H
#define CSR_COMMON_H

#include <stdint.h>

#include <omp.h>
#include <cstring>
#include "immintrin.h"
#include <math.h>

#include "../common.h"
#include "../utils.h"
#include "../format.h"
#include "../utime.h"

#include "mkl.h"

void MKL_MUL_MKL(MKLMatrix *A_csr, MKLMatrix *B_csr, MKLMatrix *C_csr)
{

    sparse_matrix_t        csrA = NULL, csrB = NULL, csrC = NULL;

    MKL_INT *columns_C = NULL; 
    struct matrix_descr    descr_type_gen;
    struct matrix_descr    descr_type_gen_A;
    struct matrix_descr    descr_type_gen_B;
    struct matrix_descr    descr_type_gen_C;

    descr_type_gen.type = SPARSE_MATRIX_TYPE_GENERAL;
    descr_type_gen_A.type = SPARSE_MATRIX_TYPE_GENERAL;
    descr_type_gen_B.type = SPARSE_MATRIX_TYPE_GENERAL;
    descr_type_gen_C.type = SPARSE_MATRIX_TYPE_GENERAL;

    sparse_index_base_t    indexing;

    mkl_sparse_d_create_csr( &csrA, SPARSE_INDEX_BASE_ZERO, A_csr->row, A_csr->col, A_csr->row_ind, A_csr->row_ind+1, A_csr->col_ind, A_csr->values );
    mkl_sparse_d_create_csr( &csrB, SPARSE_INDEX_BASE_ZERO, B_csr->row, B_csr->col, B_csr->row_ind, B_csr->row_ind+1, B_csr->col_ind, B_csr->values );

    //mkl_sparse_spmm( SPARSE_OPERATION_NON_TRANSPOSE, csrA, csrB, &csrC );
    mkl_sparse_sp2m( SPARSE_OPERATION_NON_TRANSPOSE, descr_type_gen_A, csrA, SPARSE_OPERATION_NON_TRANSPOSE, descr_type_gen_B, csrB, SPARSE_STAGE_FULL_MULT, &csrC );

    mkl_sparse_d_export_csr( csrC, &indexing, &C_csr->row, &C_csr->col, &C_csr->row_ind, &columns_C, &C_csr->col_ind, &C_csr->values );

    C_csr->nnz=C_csr->row_ind[C_csr->row];


}




void Transpose_CSR(CsrMatrix *A, CsrMatrix *A_t)
{
    A_t->row=A->col;
    A_t->col=A->row;
    A_t->nnz=A->nnz;

    A_t->row_ind=(int*)malloc1d((A_t->row+1),sizeof(int));
    A_t->col_ind=(int*)malloc1d((A_t->nnz),sizeof(int));
    A_t->values=(VALUE_TYPE*)malloc1d((A_t->nnz),sizeof(VALUE_TYPE));

    int count=0;
    A_t->row_ind[0]=count;
    int k,i,j;
    for(k=0;k<A_t->row;k++)
    {
        for(i=0;i<A->row;i++)
        {
            for(j=A->row_ind[i];j<A->row_ind[i+1];j++)
            {
                if(A->col_ind[j]==(k))
                {
                    A_t->col_ind[count]=i;
                    A_t->values[count]=A->values[j];
                    count++;
                }
            }
        }
        A_t->row_ind[k+1]=count;
    }

}


void CSR_MUL_CSR(CsrMatrix* A, CsrMatrix* B, CsrMatrix* C)
{

    C->row=A->row;
    C->col=B->col;
    C->nnz=0;
    C->row_ind=(int *)malloc((C->row+1)*sizeof(int));

    C->row_ind[0]=0;

    #pragma omp parallel
    {
    int* mask=(int *)malloc((C->col)*sizeof(int));
    for(int i=0;i<C->col;i++) mask[i]=-1;

    #pragma omp for
    for(int i = 0; i < C->row; i++)
    {
        int num_nonzeros = 0;
        for(int jj = A->row_ind[i]; jj < A->row_ind[i + 1]; jj++)
        {
            int j = A->col_ind[jj];
            for(int kk = B->row_ind[j]; kk < B->row_ind[j + 1]; kk++)
            {
                int k = B->col_ind[kk];
                if(mask[k] != i)
                    {
                        mask[k] = i;
                        num_nonzeros++;
                    } 
            }

        }
		C->row_ind[i + 1] = num_nonzeros;
    }

    }
	for(int i = 1; i <= C->row; i++)
		C->row_ind[i] += C->row_ind[i - 1];	

    C->nnz=C->row_ind[C->row];

    C->col_ind=(int *)malloc(C->nnz*sizeof(int));
    C->values=(VALUE_TYPE *)malloc(C->nnz*sizeof(VALUE_TYPE));

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

            int jj_start = A->row_ind[i];
            int jj_end   = A->row_ind[i + 1];

            for (int jj = jj_start; jj < jj_end; jj++)
            {
                int j = A->col_ind[jj];
                VALUE_TYPE v = A->values[jj];

                int kk_start = B->row_ind[j];
                int kk_end   = B->row_ind[j + 1];

                for (int kk = kk_start; kk < kk_end; kk++)
                {
                    int k = B->col_ind[kk];
                    VALUE_TYPE b = B->values[kk];

                    sums[k] = sums[k] + v * b ;

                    if (next[k] == unseen)
                    {
                        next[k] = head;
                        head = k;
                        length++;
                    }
                }
            }

            int offset = C->row_ind[i];

            for (int jj = 0; jj < length; jj++)
            {
                C->col_ind[offset] = head;
                C->values[offset] = sums[head];
                offset++;

                int temp = head;
                head = next[head];

                // clear arrays
                next[temp] = unseen;
                sums[temp] = 0.0;
            }
        } // end for loop
    } //omp parallel
 


}


double sizeofcsr(CsrMatrix *A_csr)
{
    double csr_size=0;
    csr_size+=(sizeof(int))*(A_csr->row+1+A_csr->nnz+3);
    csr_size+=(sizeof(VALUE_TYPE))*(A_csr->nnz);
    return csr_size;
}

double sizeofcsr(MKLMatrix *A_csr)
{
    double csr_size=0;
    csr_size+=(sizeof(int))*(A_csr->row+1+A_csr->nnz+3);
    csr_size+=(sizeof(VALUE_TYPE))*(A_csr->nnz);
    return csr_size;
}


void print_csr(CsrMatrix *A_csr)
{
    printf("row:%d col:%d nnz:%d\n",A_csr->row,A_csr->col,A_csr->nnz);
    for(int i=0;i<=A_csr->row;i++)
    {
        printf("%d,",A_csr->row_ind[i]);
    }
    printf("\n");

    for(int i=0;i<A_csr->nnz;i++)
    {
        printf("%d,",A_csr->col_ind[i]);
    }
    printf("\n");

    for(int i=0;i<A_csr->nnz;i++)
    {
        printf("%.2lf,",A_csr->values[i]);
    }
    printf("\n");
}

void print_csr(MKLMatrix *A_csr)
{
    printf("row:%d col:%d nnz:%d\n",A_csr->row,A_csr->col,A_csr->nnz);
    for(int i=0;i<=A_csr->row;i++)
    {
        printf("%d,",A_csr->row_ind[i]);
    }
    printf("\n");

    for(int i=0;i<A_csr->nnz;i++)
    {
        printf("%d,",A_csr->col_ind[i]);
    }
    printf("\n");

    for(int i=0;i<A_csr->nnz;i++)
    {
        printf("%.2lf,",A_csr->values[i]);
    }
    printf("\n");
}

void GetInfo1(CsrMatrix *A, double* features)
{
    int param1 = A->row;
    int param2 = A->col;
    int param3 = A->nnz;
    double param4 = (A->nnz+0.0)/(A->row*A->col);
    int param5 = A->row_ind[1]-A->row_ind[0];
    int param6 = A->row_ind[1]-A->row_ind[0];
    double param7 = (A->nnz+0.0)/A->row;
    double param8 = 0;
    for(int i=0;i<A->row;i++)
    {
        int nnz_row=A->row_ind[i+1]-A->row_ind[i];
        param5 = param5>nnz_row?param5:nnz_row;
        param6 = param6<nnz_row?param6:nnz_row;
        param8 += ((nnz_row-param7)*(nnz_row-param7));
    }
    param8 = param8/(A->row-1);
    double param9 = sqrt(param8)/param7;
    //printf("%d,%d,%d,%lf,%d,%d,%lf,%lf,%lf,",param1,param2,param3,param4,param5,param6,param7,param8,param9);
    features[0]=param1;
    features[1]=param2;
    features[2]=param3;
    features[3]=param4;
    features[4]=param5;
    features[5]=param6;
    features[6]=param7;
    features[7]=param8;
    features[8]=param9;

}


long long GetFlop(CsrMatrix *A, CsrMatrix *B)
{

    long long flops = 0.0;
    for(int i = 0; i < A->row; i++)
    {
        for(int jj = A->row_ind[i]; jj < A->row_ind[i + 1]; jj++)
        {
            int j = A->col_ind[jj];
            flops+=(B->row_ind[j + 1]-B->row_ind[j]);
        }
    }
    return flops;

}


void FreeCsrMatrix(CsrMatrix *A_csr)
{
    free(A_csr->row_ind);
    free(A_csr->col_ind);
    free(A_csr->values);
    A_csr->row=0;
    A_csr->col=0;
    A_csr->nnz=0;
    //free(A_csr);
    A_csr=NULL;
}

void FreeMKLMatrix(MKLMatrix *A_mkl)
{
    mkl_free(A_mkl->row_ind);
    mkl_free(A_mkl->col_ind);
    mkl_free(A_mkl->values);
    A_mkl->row=0;
    A_mkl->col=0;
    A_mkl->nnz=0;
    //free(A_csr);
    A_mkl=NULL;
}




#endif // CSR_COMMON_H
