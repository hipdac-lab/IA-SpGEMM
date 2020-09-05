#ifndef DIA_COMMON_H
#define DIA_COMMON_H

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

double sizeofdia(DiaMatrix *A_dia)
{
    double dia_size=0;
    dia_size+=(sizeof(int))*(A_dia->row+A_dia->col-1+A_dia->num_diagonals+3);
    dia_size+=(sizeof(VALUE_TYPE))*(A_dia->row*A_dia->num_diagonals);
    return dia_size;
}


void CSRtoDIA(CsrMatrix *A, DiaMatrix *A_dia)
{

    int num_diagonals = 0;

    int* diag_map=(int*)malloc1d((A->row+A->col),sizeof(int));

    for(int i = 0; i < A->row; i++)
    {
        for(int jj = A->row_ind[i]; jj < A->row_ind[i+1]; jj++)
        {
            int j = A->col_ind[jj];
            int map_index = (A->row - i) + j; //第几条对角线

            if(diag_map[map_index] == 0)
            {
                diag_map[map_index] = 1;
                num_diagonals++;
            }
        }
    }

    A_dia->row=A->row;
    A_dia->col=A->col;
    A_dia->num_diagonals=num_diagonals;

    A_dia->choice=true;
    if( sizeofdia(A_dia)<(50*sizeofcsr(A)))
    {

    A_dia->diagonal_ind=(int*)malloc1d((A->row+A->col-1),sizeof(int));
    A_dia->diagonal_offsets=(int*)malloc1d((A_dia->num_diagonals),sizeof(int));
    A_dia->values=(VALUE_TYPE**)malloc2d(A_dia->row,A_dia->num_diagonals,sizeof(VALUE_TYPE));
    // fill in diagonal_offsets array
    for(int n = 0, diag = 0; n < (A->row+A->col); n++)
    {
        if(diag_map[n] == 1)
        {
            diag_map[n] = diag;
            A_dia->diagonal_offsets[diag] = n - A->row;
            diag++;
        }
    }

    for(int i = 0; i < A->row; i++)
    {
        for(int jj = A->row_ind[i]; jj < A->row_ind[i+1]; jj++)
        {
            int j = A->col_ind[jj];
            int map_index = (A->row - i) + j; //offset shifted by + num_rows
            int diag = diag_map[map_index];

            A_dia->values[i][diag] = A->values[jj];
        }
    }


    for(int n = 1 ; n < (A->row+A->col); n++)
        A_dia->diagonal_ind[n-1]=diag_map[n];
    }
    else
    {
        A_dia->choice=false;
    }


    free(diag_map);
}




void DIA_mul_DIA(DiaMatrix *A_dia, DiaMatrix *B_dia, DiaMatrix *C_dia)
{

    bool* bits_flag=(bool*)malloc1d((A_dia->row+B_dia->col-1),sizeof(bool));
    int output_dia=0;

    #pragma omp for
    for(int i=0;i<A_dia->row;i++)
    {
        for(int j=0;j<A_dia->num_diagonals;j++)
        {
            int A_i=i;
            int A_j=i+A_dia->diagonal_offsets[j];
            if( A_j>=0 && A_j<A_dia->col )
            {
                for(int k=0;k<B_dia->num_diagonals;k++)
                {
                    int B_i=A_j;
                    int B_j=B_i+B_dia->diagonal_offsets[k];
                    if( B_j>=0 && B_j<B_dia->col )
                    {
                        int out_i=A_i;
                        int out_j=B_j;
                        int out_dia=A_dia->row-out_i+out_j-1;
                        if(bits_flag[out_dia]==false)
                        {
                            //omp_set_lock(&lock);
                            output_dia++;
                            bits_flag[out_dia]=true;
                            //omp_unset_lock(&lock);
                        }
                    }

                }

            }

        }

    }

    C_dia->row=A_dia->row;
    C_dia->col=B_dia->col;
    C_dia->num_diagonals=output_dia;
    C_dia->diagonal_ind=(int*)malloc1d((A_dia->row+B_dia->col-1),sizeof(int));
    C_dia->diagonal_offsets=(int*)malloc1d(C_dia->num_diagonals,sizeof(int));
    C_dia->values=(VALUE_TYPE**)malloc2d(C_dia->row,C_dia->num_diagonals,sizeof(VALUE_TYPE));


    for(int i=0,j=0;i<(A_dia->row+B_dia->col-1);i++)
    {
        if(bits_flag[i]==true)
        {
            C_dia->diagonal_ind[i]=j;
            C_dia->diagonal_offsets[j]=i+1-C_dia->row;
            j++;
        }
    }

    free(bits_flag);

    #pragma omp for
    for(int i=0;i<A_dia->row;i++)
    {
        for(int j=0;j<A_dia->num_diagonals;j++)
        {
            int A_i=i;
            int A_j=i+A_dia->diagonal_offsets[j];
            if( A_j>=0 && A_j<A_dia->col )
            {
                //printf("A:%d-%d %lf\n",A_i,A_j,A_dia->values[i][j]);
                for(int k=0;k<B_dia->num_diagonals;k++)
                {
                    int B_i=A_j;
                    int B_j=B_i+B_dia->diagonal_offsets[k];
                    if( B_j>=0 && B_j<B_dia->col )
                    {
                        //printf("B:%d-%d %lf\n",B_i,B_j,B_dia->values[B_i][k]);
                        int out_i=A_i;
                        int out_j=B_j;
                        out_j=C_dia->diagonal_ind[out_j-out_i+C_dia->row-1];
                        //printf("C:%d-%d\n",out_i,out_j);
                        C_dia->values[out_i][out_j]+=A_dia->values[i][j]*B_dia->values[B_i][k];

                    }

                }

            }

        }

    }

}



void print_dia(DiaMatrix *A_dia)
{

    printf("row:%d col:%d num_diagonals:%d\n",A_dia->row,A_dia->col,A_dia->num_diagonals);

    for(int i=0;i<(A_dia->row+A_dia->col-1);i++)
        printf("%d,",A_dia->diagonal_ind[i]);
    printf("\n");

    for(int i=0;i<A_dia->num_diagonals;i++)
        printf("%d,",A_dia->diagonal_offsets[i]);
    printf("\n");

    for(int i=0;i<A_dia->row;i++)
    {
        for(int j=0;j<A_dia->num_diagonals;j++)
            printf("%5.2lf,",A_dia->values[i][j]);
        printf("\n");
    }

}


void GetInfo2(DiaMatrix *A, double* features)
{

    int param1 = A->num_diagonals;
    double param2 = (A->num_diagonals+0.0)/(A->row+A->col-1);
    double param3 = (A->num_diagonals*A->row+0.0)/(A->row*A->col);
    //printf("%d,%lf,%lf,",param1,param2,param3);
    features[0]=param1;
    features[1]=param2;
    features[2]=param3;

}


void FreeDiaMatrix(DiaMatrix *A)
{
    free2d(A->values);
    free(A->diagonal_ind);
    free(A->diagonal_offsets);
    A->row=0;
    A->col=0;
    A->num_diagonals=0;
    A->values=NULL;
    A->diagonal_ind=NULL;
    A->diagonal_offsets=NULL;
    A=NULL;
    //free(A);
}



#endif // DIA_COMMON_H

