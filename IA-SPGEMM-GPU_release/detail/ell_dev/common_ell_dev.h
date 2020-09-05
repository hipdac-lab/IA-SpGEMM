#ifndef ELL_DEV_COMMON_H
#define ELL_DEV_COMMON_H

#include "../common.h"
#include "../utils.h"
#include "../format.h"
#include "../utime.h"





__global__ void print_dev_cu1(int num, int *dev)
{

    for(int i=0;i<num;i++)
    {
        printf("%d,",dev[i]);
    }
    printf("\n");

}

void print_dev1(int num, int *dev)
{

print_dev_cu1<<<1,1>>>(num,dev);

}

















__global__ void print_ell_cu(int row, int col, int max_nnz_per_row, int nnz, int* nnz_row, int* col_ind, double* values)
{

    printf("row:%d col:%d max_nnz_per_row:%d nnz:%d\n",row,col,max_nnz_per_row,nnz);

    printf("nnz_row:\n");
    for(int i=0;i<(row);i++)
        printf("%d,",nnz_row[i]);
    printf("\n");

    printf("col_ind:\n");
    for(int i=0;i<(row);i++)
    {
    for(int j=0;j<(max_nnz_per_row);j++)
        printf("%d,",col_ind[i*max_nnz_per_row+j]);
    printf("\n");
    }

    printf("values:\n");
    for(int i=0;i<(row);i++)
    {
    for(int j=0;j<(max_nnz_per_row);j++)
        printf("%5.2lf,",values[i*max_nnz_per_row+j]);
    printf("\n");
    }


}

void print_ell_dev(EllMatrixDev *A)
{

print_ell_cu<<<1,1>>>(A->row, A->col, A->max_nnz_per_row, A->nnz, A->nnz_row_dev, A->col_ind_dev, A->values_dev);

}




void UploadEllMatrix(EllMatrix* A_ell, EllMatrixDev* A_ell_dev)
{
A_ell_dev->row=A_ell->row;
A_ell_dev->col=A_ell->col;
A_ell_dev->max_nnz_per_row=A_ell->max_nnz_per_row;
A_ell_dev->nnz=A_ell->nnz;

DevMalloc((void**)&A_ell_dev->nnz_row_dev,sizeof(int)*(A_ell_dev->row));
DevMalloc((void**)&A_ell_dev->col_ind_dev,sizeof(int)*(A_ell_dev->row*A_ell_dev->max_nnz_per_row));
DevMalloc((void**)&A_ell_dev->values_dev,sizeof(double)*(A_ell_dev->row*A_ell_dev->max_nnz_per_row));

DevUpload(A_ell_dev->nnz_row_dev, A_ell->nnz_row, sizeof(int)*(A_ell_dev->row));
DevUpload(A_ell_dev->col_ind_dev, A_ell->col_ind, sizeof(int)*(A_ell_dev->row*A_ell_dev->max_nnz_per_row));
DevUpload(A_ell_dev->values_dev, A_ell->values, sizeof(double)*(A_ell_dev->row*A_ell_dev->max_nnz_per_row));

}

/*

__global__ void CSR_MUL_CSR_DEV_phase1(int A_row, int A_col, int B_col, int* A_row_ind, int* A_col_ind, int* B_row_ind, int* B_col_ind, int* C_row_ind, int* mask, int threads_num)
{

    int index = blockDim.x * blockIdx.x + threadIdx.x;

    for(int i=index;(i<A_row);i+=threads_num)
    {
        int num_nonzeros = 0;
        for(int jj = A_row_ind[i]; jj < A_row_ind[i + 1]; jj++)
        {
            int j = A_col_ind[jj];
            for(int kk = B_row_ind[j]; kk < B_row_ind[j + 1]; kk++)
            {
                int k = B_col_ind[kk];
                if(mask[index*B_col+k] != (i+1))
                    {
                        mask[index*B_col+k] = i+1;
                        num_nonzeros++;
                    } 
            }

        }
		C_row_ind[i + 1] = num_nonzeros;
    }

}


__global__ void CSR_MUL_CSR_DEV_upper(int A_row, int* A_row_ind, int* A_col_ind, int* B_row_ind, int* row_upper, int threads_upper)
{

    int index = blockDim.x * blockIdx.x + threadIdx.x;
    for(int i=index;i<A_row;i+=threads_upper)
    {
            int jj_start = A_row_ind[i];
            int jj_end   = A_row_ind[i + 1];
            int nnz_row=0;
            for (int jj = jj_start; jj < jj_end; jj++)
            {
                int j = A_col_ind[jj];
                int nnzs=(B_row_ind[j + 1]-B_row_ind[j]);
                nnz_row+=nnzs;
                //upper[jj]+=nnzs;
            }
            row_upper[i+1]=nnz_row;
    }
}


__global__ void CSR_MUL_CSR_DEV_Exclusive_Sum(int C_row, int* upper)
{

    for(int i = 1; i <= C_row; i++)
    {
        upper[i] += upper[i-1];   
    }

}





*/

__global__ void ELL_MUL_ELL_DEV_upper(int A_row, int A_max_nnz_per_row, int* A_nnz_row, int* A_col_ind, int* B_nnz_row, int* row_upper, int threads_upper)
{

    int index = blockDim.x * blockIdx.x + threadIdx.x;
    for(int i=index;i<A_row;i+=threads_upper)
    {
            int count = A_nnz_row[i];
            int nnz_row=0;
            for (int jj = 0; jj < count; jj++)
            {
                int j = A_col_ind[i*A_max_nnz_per_row+jj];
                int nnzs=B_nnz_row[j];
                nnz_row+=nnzs;
                //upper[jj]+=nnzs;
            }
            row_upper[i]=nnz_row;
    }
}


__global__ void ELL_MUL_ELL_DEV_Exclusive_Scan(int C_row, int* nnz_dev, int* max_nnz_per_row)
{
    max_nnz_per_row[0]=0;
    for(int i = 0; i < C_row; i++)
    {
        max_nnz_per_row[0]=(max_nnz_per_row[0]>nnz_dev[i]?max_nnz_per_row[0]:nnz_dev[i]);
    }

}


__global__ void ELL_MUL_ELL_DEV_Exclusive_Sum(int C_row, int* nnz_dev, int* max_nnz_per_row, int* nnz)
{
    nnz[0]=0;
    max_nnz_per_row[0]=0;
    for(int i = 0; i < C_row; i++)
    {
        max_nnz_per_row[0]=(max_nnz_per_row[0]>nnz_dev[i]?max_nnz_per_row[0]:nnz_dev[i]);
        nnz[0]+=nnz_dev[i];
    }

}


__global__ void ELL_MUL_ELL_DEV_phase2(int A_row, int A_max_nnz_per_row, int B_max_nnz_per_row, int* A_nnz_row, int* A_col_ind, double* A_values, int* B_nnz_row, int* B_col_ind, double* B_values, int C_max_nnz_per_row, int* C_col_ind, double* C_values, int threads_num)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;

    int unseen = -1;
    int init = -100;

    for(int i=index;i<A_row;i+=threads_num)
        {

            int count_A = A_nnz_row[i];
            int offsets=0;
            for (int jj = 0; jj < count_A; jj++)
            {
                int j = A_col_ind[i*A_max_nnz_per_row+jj];
                double v = A_values[i*A_max_nnz_per_row+jj];

                int count_B=B_nnz_row[j];

                for (int kk = 0; kk < count_B; kk++)
                {
                    int k = B_col_ind[j*B_max_nnz_per_row+kk];
                    double b = B_values[j*B_max_nnz_per_row+kk];

                    C_col_ind[i*C_max_nnz_per_row+offsets] = k;
                    C_values[i*C_max_nnz_per_row+offsets] = v * b ;
                    offsets++;
                }
            }

        } // end for loop


}



__global__ void ELL_MUL_ELL_DEV_phase3(int C_row, int C_max_nnz_per_row, int* row_offset, int* nnz_row, int* C_col_ind, double* C_values, int threads_num)
{

    int index = blockDim.x * blockIdx.x + threadIdx.x;

    for(int i=index;i<C_row;i+=threads_num)
    {
        //int count=row_offset[i];
        int count=C_max_nnz_per_row;
        for(int j=0;j<count;j++)
        {
            int offset = C_col_ind[i*C_max_nnz_per_row+j];
            for(int w=j+1;w<count;w++)
            {
                if(offset>=C_col_ind[i*C_max_nnz_per_row+w])
                {
                    C_values[i*C_max_nnz_per_row+j]+=C_values[i*C_max_nnz_per_row+w];
                    C_col_ind[i*C_max_nnz_per_row+w]=-1;
                    C_values[i*C_max_nnz_per_row+w]=0;
                }
            }

        }
        int nnz_real=0;
        for(int j=0;j<count;j++)
        {
            if(C_col_ind[i*C_max_nnz_per_row+j]!=-1)
                nnz_real++;
        }
        nnz_row[i]=nnz_real;
    }

}


__global__ void ELL_MUL_ELL_DEV_phase4(int C_row, int max_nnz_per_row_temp, int C_max_nnz_per_row, int* C_nnz_row, int* C_col_ind, double* C_values, int* col_ind, double* values, int threads_num_phase4)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;

    for(int i=index;i<C_row;i+=threads_num_phase4)
    {
        int offset=0;
        for(int j=0;j<max_nnz_per_row_temp;j++)
        {
            if(col_ind[i*max_nnz_per_row_temp+j]!=-1)
            {
            C_col_ind[i*C_max_nnz_per_row+offset]=col_ind[i*max_nnz_per_row_temp+j];
            C_values[i*C_max_nnz_per_row+offset]=values[i*max_nnz_per_row_temp+j];
            offset++;
            }
        }
    }


}




double ELL_MUL_ELL_DEV(EllMatrixDev* A, EllMatrixDev* B, EllMatrixDev* C, anonymouslib_timer_gpu* ref_timer_gpu)
{
    int *row_ind_dev;
    int *col_ind_dev;
    int *max_nnz_per_row;
    int max_nnz_per_row_temp;
    int *nnz;
    double *values_dev;
    int threads_upper=1024;
    int threads_num_phase1=1024;
    int threads_num_phase2=1024;
    int threads_num_phase3=1024;
    int threads_num_phase4=1024;
    C->row=A->row;
    C->col=B->col;

    int* row_offset;

    DevMalloc((void**)&max_nnz_per_row,sizeof(int));
    DevMalloc((void**)&nnz,sizeof(int));
    DevMalloc((void**)&row_ind_dev,sizeof(int)*(C->row+1));
    DevMalloc((void**)&row_offset,sizeof(int)*(C->row));
    DevMalloc((void**)&C->nnz_row_dev,sizeof(int)*(C->row));

    ref_timer_gpu->start();

    ELL_MUL_ELL_DEV_upper<<<threads_upper/4,4>>>(A->row, A->max_nnz_per_row, A->nnz_row_dev, A->col_ind_dev, B->nnz_row_dev, row_offset, threads_upper);

    ELL_MUL_ELL_DEV_Exclusive_Scan<<<1,1>>>(C->row, row_offset, max_nnz_per_row);

    DevDownload(&C->max_nnz_per_row, max_nnz_per_row, sizeof(int));

    max_nnz_per_row_temp=C->max_nnz_per_row;

    //print_dev1(C->row,row_offset);
    //print_dev1(1,max_nnz_per_row);

    DevMalloc((void**)&col_ind_dev,sizeof(int)*(C->row*C->max_nnz_per_row));
    DevMalloc((void**)&values_dev,sizeof(double)*(C->row*C->max_nnz_per_row));


	ELL_MUL_ELL_DEV_phase2<<<threads_num_phase2/4,4>>>(A->row, A->max_nnz_per_row, B->max_nnz_per_row, A->nnz_row_dev, A->col_ind_dev, A->values_dev, B->nnz_row_dev, B->col_ind_dev, B->values_dev, C->max_nnz_per_row, col_ind_dev, values_dev, threads_num_phase2);

    //print_dev1(C->row*C->max_nnz_per_row,col_ind_dev);
    //printd_dev(C->row*C->max_nnz_per_row,values_dev);

	ELL_MUL_ELL_DEV_phase3<<<threads_num_phase3/4,4>>>(C->row, C->max_nnz_per_row, row_offset, C->nnz_row_dev, col_ind_dev, values_dev, threads_num_phase3);

    //print_dev1(C->row,C->nnz_row_dev);
    //print_dev1(C->row*C->max_nnz_per_row,col_ind_dev);
    //printd_dev(C->row*C->max_nnz_per_row,values_dev);

    ELL_MUL_ELL_DEV_Exclusive_Sum<<<1,1>>>(C->row, C->nnz_row_dev, max_nnz_per_row, nnz);
    DevDownload(&C->max_nnz_per_row, max_nnz_per_row, sizeof(int));
    DevDownload(&C->nnz, nnz, sizeof(int));

    DevMalloc((void**)&C->col_ind_dev,sizeof(int)*(C->row*C->max_nnz_per_row));
    DevMalloc((void**)&C->values_dev,sizeof(double)*(C->row*C->max_nnz_per_row));

	ELL_MUL_ELL_DEV_phase4<<<threads_num_phase4/4,4>>>(C->row, max_nnz_per_row_temp, C->max_nnz_per_row, C->nnz_row_dev, C->col_ind_dev, C->values_dev, col_ind_dev, values_dev, threads_num_phase4);

    //print_dev1(C->row,C->nnz_row_dev);
    //print_dev1(C->row*C->max_nnz_per_row,C->col_ind_dev);
    //printd_dev(C->row*C->max_nnz_per_row,C->values_dev);

    double time_ref=ref_timer_gpu->stop();

    cudaFree(row_offset);
    cudaFree(col_ind_dev);
    cudaFree(values_dev);
    return time_ref;

}




__global__ void getsum_ell_cu(double* sum, int row, int max_nnz_per_row, double* values)
{
sum[0]=0.0;
for(int i=0;i<row;i++)
for(int j=0;j<max_nnz_per_row;j++)
    sum[0]+=values[i*max_nnz_per_row+j];
}

double getsum_ell(EllMatrixDev *A)
{
double sum;
double *sum_dev;
DevMalloc((void**)&sum_dev,sizeof(double));
getsum_ell_cu<<<1,1>>>(sum_dev, A->row, A->max_nnz_per_row, A->values_dev);
DevDownload(&sum, sum_dev, sizeof(double));
cudaFree(sum_dev);
return sum;
}

double sizeofell(EllMatrixDev *A_ell_dev)
{
    double ell_size=0;
    ell_size+=(sizeof(int))*(A_ell_dev->row+A_ell_dev->row*A_ell_dev->max_nnz_per_row+4);
    ell_size+=(sizeof(double))*((A_ell_dev->row*A_ell_dev->max_nnz_per_row));
    return ell_size;
}










void FreeEllMatrixDev(EllMatrixDev *A)
{
    cudaFree(A->nnz_row_dev);
    cudaFree(A->col_ind_dev);
    cudaFree(A->values_dev);
    A->row=0;
    A->col=0;
    A->max_nnz_per_row=0;
    A->nnz=0;
    //free(A_csr);
    A=NULL;
}


#endif // ELL_DEV_COMMON_H
