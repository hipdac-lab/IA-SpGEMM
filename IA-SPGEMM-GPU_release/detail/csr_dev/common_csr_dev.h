#ifndef CSR_DEV_COMMON_H
#define CSR_DEV_COMMON_H

#include "../common.h"
#include "../utils.h"
#include "../format.h"
#include "../utime.h"

#include <thrust/scan.h>
#include <thrust/execution_policy.h>

#include <thrust/functional.h>
#include <thrust/iterator/permutation_iterator.h>

#include <cusp/sort.h>
#include <cusp/format_utils.h>



__global__ void print_dev_cu(int num, int *dev)
{

    for(int i=0;i<num;i++)
    {
        printf("%d,",dev[i]);
    }
    printf("\n");

}

void print_dev(int num, int *dev)
{

print_dev_cu<<<1,1>>>(num,dev);

}




__global__ void printd_dev_cu(int num, double *dev)
{

    for(int i=0;i<num;i++)
    {
        printf("%.2lf,",dev[i]);
    }
    printf("\n");

}

void printd_dev(int num, double *dev)
{

printd_dev_cu<<<1,1>>>(num,dev);

}



__global__ void csr_to_coo(int row, int* row_ind_dev, int* A_coo)
{

for(int i=0;i<row;i++)
{
for(int j=row_ind_dev[i];j<row_ind_dev[i+1];j++)
A_coo[j]=i;
}

}





__global__ void print_csr_cu(int row, int col, int nnz, int* row_ind, int* col_ind, double* values)
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

void print_csr_dev(CsrMatrixDev *A)
{

print_csr_cu<<<1,1>>>(A->row, A->col, A->nnz, A->row_ind_dev, A->col_ind_dev, A->values_dev);

}





void UploadCsrMatrix(CsrMatrix* A_csr, CsrMatrixDev* A_csr_dev)
{
A_csr_dev->row=A_csr->row;
A_csr_dev->col=A_csr->col;
A_csr_dev->nnz=A_csr->nnz;

DevMalloc((void**)&A_csr_dev->row_ind_dev,sizeof(int)*(A_csr_dev->row+1));
DevMalloc((void**)&A_csr_dev->col_ind_dev,sizeof(int)*(A_csr_dev->nnz));
DevMalloc((void**)&A_csr_dev->values_dev,sizeof(double)*(A_csr_dev->nnz));

DevUpload(A_csr_dev->row_ind_dev, A_csr->row_ind, sizeof(int)*(A_csr_dev->row+1));
DevUpload(A_csr_dev->col_ind_dev, A_csr->col_ind, sizeof(int)*(A_csr_dev->nnz));
DevUpload(A_csr_dev->values_dev, A_csr->values, sizeof(double)*(A_csr_dev->nnz));

}


__global__ void phase1(int nnz, int* segment_lengths, int* output_ptr)
{
output_ptr[nnz] = output_ptr[nnz - 1] + segment_lengths[nnz - 1];
}


double CSR_MUL_CSR_DEV(CsrMatrixDev* A, CsrMatrixDev* B, CsrMatrixDev* C, anonymouslib_timer_gpu* ref_timer_gpu)
{

    ref_timer_gpu->start();

    C->row=A->row;
    C->col=B->col;

	int* A_coo;
    DevMalloc((void**)&A_coo,sizeof(int)*(A->row));
    csr_to_coo<<<1,1>>>(A->row,A->row_ind_dev,A_coo);

    int* B_row_length;
    DevMalloc((void**)&B_row_length,sizeof(int)*(B->row));
    thrust::transform(thrust::device, &B->row_ind_dev[1], &B->row_ind_dev[B->row+1], &B->row_ind_dev[0], &B_row_length[0], thrust::minus<int>());

    int* segment_lengths;
    DevMalloc((void**)&segment_lengths,sizeof(int)*(A->nnz));
    thrust::gather(thrust::device, A->col_ind_dev, &A->col_ind_dev[A->nnz], B_row_length, segment_lengths);

    int* output_ptr;
    DevMalloc((void**)&output_ptr,sizeof(int)*(A->nnz+1));
    thrust::exclusive_scan(thrust::device, segment_lengths, &segment_lengths[A->nnz], output_ptr, int(0));

    phase1<<<1,1>>>(A->nnz,segment_lengths,output_ptr);

    int coo_num_nonzeros;
    DevDownload(&coo_num_nonzeros, &output_ptr[A->nnz], sizeof(int));

	int workspace_capacity = thrust::min<int>(coo_num_nonzeros, 16 << 20);

    {
        size_t free, total;
        cudaMemGetInfo(&free, &total);

        // divide free bytes by the size of each workspace unit
        int max_workspace_capacity = free / (4 * sizeof(int) + sizeof(double));

        // use at most one third of the remaining capacity
        workspace_capacity = thrust::min<int>(max_workspace_capacity / 3, workspace_capacity);
    }



	int* A_gather_locations;
    int* B_gather_locations;
    int* I;
    int* J;
    double* V;

		int begin_row      = 0;
        int end_row        = A->row;
        int begin_segment  = 0;
        int end_segment    = A->nnz;
        int workspace_size = coo_num_nonzeros;

    DevMalloc((void**)&A_gather_locations,sizeof(int)*(workspace_size));
    DevMalloc((void**)&B_gather_locations,sizeof(int)*(workspace_size));
    DevMalloc((void**)&I,sizeof(int)*(workspace_size));
    DevMalloc((void**)&J,sizeof(int)*(workspace_size));
    DevMalloc((void**)&V,sizeof(double)*(workspace_size));

    thrust::fill(thrust::device, A_gather_locations, &A_gather_locations[workspace_size], 0);

    thrust::scatter_if(thrust::device, thrust::counting_iterator<int>(begin_segment), thrust::counting_iterator<int>(end_segment), output_ptr, segment_lengths, A_gather_locations);

    thrust::inclusive_scan(thrust::device, A_gather_locations, &A_gather_locations[workspace_size], A_gather_locations, thrust::maximum<int>());

    thrust::fill(thrust::device, B_gather_locations, &B_gather_locations[workspace_size], 1);

	thrust::scatter_if(thrust::device, thrust::make_permutation_iterator(B->row_ind_dev, A->col_ind_dev), thrust::make_permutation_iterator(B->row_ind_dev, A->col_ind_dev) + end_segment, output_ptr, segment_lengths, B_gather_locations);

    thrust::inclusive_scan_by_key(thrust::device, A_gather_locations, &A_gather_locations[workspace_size], B_gather_locations, B_gather_locations);


	thrust::gather(thrust::device,
                   A_gather_locations, &A_gather_locations[workspace_size],
                   A_coo,
                   I);
	thrust::gather(thrust::device,
                   B_gather_locations, &B_gather_locations[workspace_size],
                   B->col_ind_dev,
                   J);

    thrust::transform(thrust::device, thrust::make_permutation_iterator(A->values_dev, A_gather_locations), thrust::make_permutation_iterator(A->values_dev, &A_gather_locations[workspace_size]), thrust::make_permutation_iterator(B->values_dev, B_gather_locations), V, thrust::multiplies<double>());

    cusp::array1d_view< int* > II(I,&I[workspace_size]);
    cusp::array1d_view< int* > JJ(J,&J[workspace_size]);
    cusp::array1d_view< double* > VV(V,&V[workspace_size]);

    cusp::sort_by_row_and_column(thrust::device, II, JJ, VV);

	int NNZ = thrust::inner_product(thrust::device,
                                          thrust::make_zip_iterator(thrust::make_tuple(II.begin(), JJ.begin())),
                                          thrust::make_zip_iterator(thrust::make_tuple(II.end (),  JJ.end()))   - 1,
                                          thrust::make_zip_iterator(thrust::make_tuple(II.begin(), JJ.begin())) + 1,
                                          int(0),
                                          thrust::plus<int>(),
                                          thrust::not_equal_to< thrust::tuple<int,int> >()) + 1;

    C->nnz=NNZ;

    DevMalloc((void**)&C->row_ind_dev,sizeof(int)*(A->row+1));
    DevMalloc((void**)&C->col_ind_dev,sizeof(int)*(NNZ));
    DevMalloc((void**)&C->values_dev,sizeof(double)*(NNZ));

	thrust::reduce_by_key
    (thrust::device,
     thrust::make_zip_iterator(thrust::make_tuple(I, J)),
     thrust::make_zip_iterator(thrust::make_tuple(&I[workspace_size], &J[workspace_size])),
     V,
     thrust::make_zip_iterator(thrust::make_tuple(C->row_ind_dev, C->col_ind_dev)),
     C->values_dev,
     thrust::equal_to< thrust::tuple<int,int> >(),
     thrust::plus<double>() );

    double time_ref=ref_timer_gpu->stop();

    return time_ref;

}



__global__ void getsum_csr_cu(double* sum, int nnz, double* values)
{
for(int i=0;i<nnz;i++)
    sum[0]+=values[i];
}

double getsum_csr(CsrMatrixDev *A)
{
double sum;
double *sum_dev;
DevMalloc((void**)&sum_dev,sizeof(double));
getsum_csr_cu<<<1,1>>>(sum_dev, A->nnz, A->values_dev);
DevDownload(&sum, sum_dev, sizeof(double));
cudaFree(sum_dev);
return sum;
}

double sizeofcsr(CsrMatrixDev *A)
{
    double csr_size=0;
    csr_size+=(sizeof(int))*(A->row+1+A->nnz+3);
    csr_size+=(sizeof(double))*(A->nnz);
    return csr_size;
}



void FreeCsrMatrixDev(CsrMatrixDev *A)
{
    cudaFree(A->row_ind_dev);
    cudaFree(A->col_ind_dev);
    cudaFree(A->values_dev);
    A->row=0;
    A->col=0;
    A->nnz=0;
    //free(A_csr);
    A=NULL;
}


#endif // CSR_DEV_COMMON_H
