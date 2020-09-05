#ifndef DIA_DEV_COMMON_H
#define DIA_DEV_COMMON_H

#include "../common.h"
#include "../utils.h"
#include "../format.h"
#include "../utime.h"


void UploadDiaMatrix(DiaMatrix* A_dia, DiaMatrixDev* A_dia_dev)
{
    A_dia_dev->row=A_dia->row;
    A_dia_dev->col=A_dia->col;
    A_dia_dev->num_diagonals=A_dia->num_diagonals;

    DevMalloc((void**)&A_dia_dev->diagonal_ind_dev,sizeof(int)*(A_dia_dev->row+A_dia_dev->col-1));
    DevMalloc((void**)&A_dia_dev->diagonal_offsets_dev,sizeof(int)*(A_dia_dev->num_diagonals));
    DevMalloc((void**)&A_dia_dev->values_dev,sizeof(double)*(A_dia_dev->row)*(A_dia_dev->num_diagonals));

    DevUpload(A_dia_dev->diagonal_ind_dev, A_dia->diagonal_ind, sizeof(int)*(A_dia_dev->row+A_dia_dev->col-1));
    DevUpload(A_dia_dev->diagonal_offsets_dev, A_dia->diagonal_offsets, sizeof(int)*(A_dia_dev->num_diagonals));
    DevUpload(A_dia_dev->values_dev, A_dia->values, sizeof(double)*(A_dia_dev->row)*(A_dia_dev->num_diagonals));

}


__global__ void DIA_mul_DIA_phase1(int A_dia_row, int A_dia_col, int A_dia_num_diagonals, int* A_dia_diagonal_offsets, int B_dia_row, int B_dia_col, int B_dia_num_diagonals, int* B_dia_diagonal_offsets, bool* bits_flag, int* output_dia, int threads_num)
{
    //output_dia[0]=0;
    int index = blockDim.x * blockIdx.x + threadIdx.x;

    for(int i=index;i<A_dia_row;i+=threads_num)
    {
        for(int j=0;j<A_dia_num_diagonals;j++)
        {
            int A_i=i;
            int A_j=i+A_dia_diagonal_offsets[j];
            if( A_j>=0 && A_j<A_dia_col )
            {
                for(int k=0;k<B_dia_num_diagonals;k++)
                {
                    int B_i=A_j;
                    int B_j=B_i+B_dia_diagonal_offsets[k];
                    if( B_j>=0 && B_j<B_dia_col )
                    {
                        int out_i=A_i;
                        int out_j=B_j;
                        int out_dia=A_dia_row-out_i+out_j-1;
                        if(bits_flag[out_dia]==false)
                        {
                            //output_dia[0]++;
                            bits_flag[out_dia]=true;
                        }
                    }

                }

            }

        }

    }


}


__global__ void DIA_mul_DIA_phase2(int A_dia_row, int B_dia_col, int* C_dia_diagonal_ind, int* C_dia_diagonal_offsets, bool* bits_flag, int threads_num_phase)
{

    int index = blockDim.x * blockIdx.x + threadIdx.x;

    for(int i=index;i<(A_dia_row+B_dia_col-1);i+=threads_num_phase)
    {
        if(bits_flag[i]==true)
        {
            C_dia_diagonal_offsets[C_dia_diagonal_ind[i]]=i+1-A_dia_row;
        }
    }

}



__global__ void DIA_mul_DIA_phase3(int A_dia_row, int A_dia_col, int A_dia_num_diagonals, int* A_dia_diagonal_offsets, double* A_dia_values, int B_dia_row, int B_dia_col, int B_dia_num_diagonals, int* B_dia_diagonal_offsets, double* B_dia_values, int C_dia_row, int C_dia_num_diagonals, int* C_dia_diagonal_ind, int* C_dia_diagonal_offsets, double* C_dia_values, int threads_num)
{

    int index = blockDim.x * blockIdx.x + threadIdx.x;

    for(int i=index;i<A_dia_row;i+=threads_num)
    {
        for(int j=0;j<A_dia_num_diagonals;j++)
        {
            int A_i=i;
            int A_j=i+A_dia_diagonal_offsets[j];
            if( A_j>=0 && A_j<A_dia_col )
            {
                //printf("A:%d-%d %lf\n",A_i,A_j,A_dia->values[i][j]);
                for(int k=0;k<B_dia_num_diagonals;k++)
                {
                    int B_i=A_j;
                    int B_j=B_i+B_dia_diagonal_offsets[k];
                    if( B_j>=0 && B_j<B_dia_col )
                    {
                        //printf("B:%d-%d %lf\n",B_i,B_j,B_dia->values[B_i][k]);
                        int out_i=A_i;
                        int out_j=B_j;
                        out_j=C_dia_diagonal_ind[out_j-out_i+C_dia_row-1];
                        //printf("C:%d-%d\n",out_i,out_j);
                        C_dia_values[out_i*C_dia_num_diagonals+out_j]+=A_dia_values[i*A_dia_num_diagonals+j]*B_dia_values[B_i*B_dia_num_diagonals+k];

                    }

                }

            }

        }

    }

}


__global__ void DIA_sum(int count, bool*bits_flag, int* output_dia, int* C_dia_diagonal_ind)
{
    for(int i=0,j=0;i<count;i++)
    {
    if(bits_flag[i]==true)
    {
        output_dia[0]++;
        C_dia_diagonal_ind[i]=j;
        j++;
    }
    }
}

double DIA_MUL_DIA_DEV(DiaMatrixDev* A_dia, DiaMatrixDev* B_dia, DiaMatrixDev* C_dia, anonymouslib_timer_gpu* ref_timer_gpu)
{
    C_dia->row=A_dia->row;
    C_dia->col=B_dia->col;

    int *output_dia;
    bool* bits_flag;
    DevMalloc((void**)&bits_flag,sizeof(bool)*(A_dia->row+B_dia->col-1));
    DevMalloc((void**)&output_dia,sizeof(int));
    DevMalloc((void**)&C_dia->diagonal_ind_dev,sizeof(int)*(C_dia->row+C_dia->col-1));

    int threads_num_phase=1024;

    ref_timer_gpu->start();

    DIA_mul_DIA_phase1<<<threads_num_phase/4,4>>>(A_dia->row,A_dia->col,A_dia->num_diagonals,A_dia->diagonal_offsets_dev,B_dia->row,B_dia->col,B_dia->num_diagonals,B_dia->diagonal_offsets_dev,bits_flag,output_dia,threads_num_phase);

    //printf("time : %lf\n",ref_timer_gpu->stop());
    //ref_timer_gpu->start();

    DIA_sum<<<1,1>>>(A_dia->row+B_dia->col-1, bits_flag, output_dia, C_dia->diagonal_ind_dev);

    //printf("time : %lf\n",ref_timer_gpu->stop());
    //ref_timer_gpu->start();

    DevDownload(&C_dia->num_diagonals, output_dia, sizeof(int));

    DevMalloc((void**)&C_dia->diagonal_offsets_dev,sizeof(int)*(C_dia->num_diagonals));
    DevMalloc((void**)&C_dia->values_dev,sizeof(double)*(C_dia->num_diagonals*C_dia->row));

    DIA_mul_DIA_phase2<<<threads_num_phase/4,4>>>(A_dia->row,B_dia->col,C_dia->diagonal_ind_dev,C_dia->diagonal_offsets_dev,bits_flag,threads_num_phase);

    //printf("time : %lf\n",ref_timer_gpu->stop());
    //ref_timer_gpu->start();

    DIA_mul_DIA_phase3<<<threads_num_phase/4,4>>>(A_dia->row,A_dia->col,A_dia->num_diagonals,A_dia->diagonal_offsets_dev,A_dia->values_dev,B_dia->row,B_dia->col,B_dia->num_diagonals,B_dia->diagonal_offsets_dev,B_dia->values_dev,C_dia->row,C_dia->num_diagonals,C_dia->diagonal_ind_dev,C_dia->diagonal_offsets_dev,C_dia->values_dev,threads_num_phase);

    double time_ref=ref_timer_gpu->stop();

    cudaFree(bits_flag);
    cudaFree(output_dia);

    return time_ref;

}


void FreeDiaMatrixDev(DiaMatrixDev *A)
{
    if(A->choice)
    {
    cudaFree(A->diagonal_ind_dev);
    cudaFree(A->diagonal_offsets_dev);
    cudaFree(A->values_dev);
    }
    A->row=0;
    A->col=0;
    A->num_diagonals=0;
    //free(A_csr);
    A=NULL;
}



__global__ void print_dia_cu(int A_row, int A_col, int A_num_diagonals, int* A_diagonal_ind_dev, int* A_diagonal_offsets_dev, double* A_values_dev)
{

    printf("row:%d col:%d num_diagonals:%d\n",A_row,A_col,A_num_diagonals);

    for(int i=0;i<(A_row+A_col-1);i++)
        printf("%d,",A_diagonal_ind_dev[i]);
    printf("\n");

    for(int i=0;i<A_num_diagonals;i++)
        printf("%d,",A_diagonal_offsets_dev[i]);
    printf("\n");

    for(int i=0;i<A_row;i++)
    {
        for(int j=0;j<A_num_diagonals;j++)
            printf("%5.2lf,",A_values_dev[i*A_num_diagonals+j]);
        printf("\n");
    }

}


void print_dia_dev(DiaMatrixDev *A)
{

    print_dia_cu<<<1,1>>>(A->row, A->col, A->num_diagonals, A->diagonal_ind_dev, A->diagonal_offsets_dev, A->values_dev);

}




__global__ void getsum_dia_cu(double* sum, int row, int dia_num, double* values)
{
for(int i=0;i<row;i++)
for(int j=0;j<dia_num;j++)
    sum[0]+=values[i*dia_num+j];
}

double getsum_dia(DiaMatrixDev *A)
{
double sum;
double *sum_dev;
DevMalloc((void**)&sum_dev,sizeof(double));
getsum_dia_cu<<<1,1>>>(sum_dev, A->row, A->num_diagonals, A->values_dev);
DevDownload(&sum, sum_dev, sizeof(double));
cudaFree(sum_dev);
return sum;
}

double sizeofdia(DiaMatrixDev *A_dia_dev)
{
    double dia_size=0;
    dia_size+=(sizeof(int))*(A_dia_dev->row+A_dia_dev->col-1+A_dia_dev->num_diagonals+3);
    dia_size+=(sizeof(double))*((A_dia_dev->row)*(A_dia_dev->num_diagonals));
    return dia_size;
}







#endif // DIA_DEV_COMMON_H
