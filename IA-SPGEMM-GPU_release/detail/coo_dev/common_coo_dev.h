#ifndef COO_DEV_COMMON_H
#define COO_DEV_COMMON_H

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




__global__ void print_coo_cu(int row, int col, int nnz, int* row_ind, int* col_ind, double* values)
{

    printf("row:%d col:%d nnz:%d\n",row,col,nnz);

    for(int i=0;i<nnz;i++)
    {
        printf("%d,%d,%5.2lf\n",row_ind[i],col_ind[i],values[i]);
    }
    printf("\n");

}

void print_coo_dev(CooMatrixDev *A)
{

print_coo_cu<<<1,1>>>(A->row, A->col, A->nnz, A->row_ind_dev, A->col_ind_dev, A->values_dev);

}



void UploadCooMatrix(CooMatrix* A_coo, CooMatrixDev* A_coo_dev)
{
A_coo_dev->row=A_coo->row;
A_coo_dev->col=A_coo->col;
A_coo_dev->nnz=A_coo->nnz;

DevMalloc((void**)&A_coo_dev->row_offset_dev,sizeof(int)*(A_coo_dev->row+1));
DevMalloc((void**)&A_coo_dev->row_ind_dev,sizeof(int)*(A_coo_dev->nnz));
DevMalloc((void**)&A_coo_dev->col_ind_dev,sizeof(int)*(A_coo_dev->nnz));
DevMalloc((void**)&A_coo_dev->values_dev,sizeof(double)*(A_coo_dev->nnz));

DevUpload(A_coo_dev->row_offset_dev, A_coo->row_offset, sizeof(int)*(A_coo_dev->row+1));
DevUpload(A_coo_dev->row_ind_dev, A_coo->row_ind, sizeof(int)*(A_coo_dev->nnz));
DevUpload(A_coo_dev->col_ind_dev, A_coo->col_ind, sizeof(int)*(A_coo_dev->nnz));
DevUpload(A_coo_dev->values_dev, A_coo->values, sizeof(double)*(A_coo_dev->nnz));

}



__global__ void COO_MUL_COO_DEV_phase1(int A_row, int A_col, int B_col, int* A_row_offset, int* A_row_ind, int* A_col_ind, int* B_row_offset, int* B_row_ind, int* B_col_ind, int* C_row_offset, int* mask)
{

    for(int i=0;i<A_row;i++)
    {
        int num_nonzeros = 0;
        for(int jj = A_row_offset[i]; jj < A_row_offset[i + 1]; jj++)
        {
            int j = A_col_ind[jj];
            for(int kk = B_row_offset[j]; kk < B_row_offset[j + 1]; kk++)
            {
                int k = B_col_ind[kk];
                if(mask[k] != (i+1))
                    {
                        mask[k] = i+1;
                        num_nonzeros++;
                    } 
            }

        }
        C_row_offset[i + 1] = num_nonzeros;
    }

}


__global__ void COO_MUL_COO_DEV_Exclusive_Scan(int C_row, int *C_row_ind)
{

    C_row_ind[0]=0;
    for(int i = 1; i <= C_row; i++)
    {
        C_row_ind[i] += C_row_ind[i-1];   
    }


}



__global__ void COO_MUL_COO_DEV_phase2(int A_row, int A_col, int B_col, int* A_row_offset, int* A_row_ind, int* A_col_ind, double* A_values, int* B_row_offset, int* B_row_ind, int* B_col_ind, double* B_values, int* C_row_offset, int* C_row_ind, int* C_col_ind, double* C_values, int threads_num)
{

    int index = blockDim.x * blockIdx.x + threadIdx.x;

    for(int k=index;k<A_row;k+=threads_num)
    {
        for(int i=A_row_offset[k];i<A_row_offset[k+1];i++)
        {
            int A_i=A_row_ind[i];
            int A_j=A_col_ind[i];
            for(int j=B_row_offset[A_j];j<B_row_offset[A_j+1];j++)
            {
                int B_i=B_row_ind[j];
                int B_j=B_col_ind[j];
                //printf("write : %d,%d\n",A_i,B_j);
                for(int pos=C_row_offset[A_i];pos<C_row_offset[A_i+1];pos++)
                {
                    if(C_row_ind[pos]==-1)
                    {
                        //printf("new : %d,%d,%d\n",A_i,B_j,pos);
                        C_row_ind[pos]=A_i;
                        C_col_ind[pos]=B_j;
                        C_values[pos]=A_values[i]*B_values[j];
                        break;
                    }
                    else if((C_row_ind[pos]==A_i)&&(C_col_ind[pos]==B_j))
                    {
                        //printf("renew : %d,%d,%d\n",A_i,B_j,pos);
                        C_values[pos]+=A_values[i]*B_values[j];
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


__global__ void COO_MUL_COO_DEV_SET1(int count, int* row_ind_dev, int threads_num)
{

    int index = blockDim.x * blockIdx.x + threadIdx.x;

    for(int i=index;i<count;i+=threads_num)
        row_ind_dev[i]=-1;
}

__global__ void coo_to_csr(int row, int nnz, int* coo, int* csr)
{

csr[0]=0;
for(int i=0;i<nnz;i++)
{
csr[coo[i]+1]++;
}

for(int i=0;i<row;i++)
{
csr[i+1]=csr[i+1]+csr[i];
}


}





void coo_spmm_helper(size_t workspace_size, size_t begin_row, size_t end_row, size_t begin_segment, size_t end_segment, CooMatrixDev *A, CooMatrixDev *B, CooMatrixDev *C, const cusp::array1d<int, cusp::device_memory>& B_row_offsets, const cusp::array1d<int, cusp::device_memory>& segment_lengths, const cusp::array1d<int, cusp::device_memory>& output_ptr, cusp::array1d<int, cusp::device_memory>& A_gather_locations, cusp::array1d<int, cusp::device_memory>& B_gather_locations, cusp::array1d<int, cusp::device_memory> & I, cusp::array1d<int, cusp::device_memory> & J, cusp::array1d<double, cusp::device_memory> & V)
{
	typedef int IndexType;
    typedef double ValueType;
    typedef typename cusp::device_memory MemorySpace;

    A_gather_locations.resize(workspace_size);
    B_gather_locations.resize(workspace_size);
    I.resize(workspace_size);
    J.resize(workspace_size);
    V.resize(workspace_size);

/*
    // nothing to do
    if (!workspace_size)
    {
        C->nnz=0;
        return;
    } 
*/

	// compute gather locations of intermediate format
    thrust::fill(thrust::device, A_gather_locations.begin(), A_gather_locations.end(), 0);
    thrust::scatter_if(thrust::device,
                       thrust::counting_iterator<IndexType>(begin_segment), thrust::counting_iterator<IndexType>(end_segment),
                       output_ptr.begin() + begin_segment,
                       segment_lengths.begin() + begin_segment,
                       A_gather_locations.begin() - output_ptr[begin_segment]);
    thrust::inclusive_scan(thrust::device, A_gather_locations.begin(), A_gather_locations.end(), A_gather_locations.begin(), thrust::maximum<IndexType>());


    // compute gather locations of intermediate format
    thrust::fill(thrust::device, B_gather_locations.begin(), B_gather_locations.end(), 1);

    cusp::array1d_view< int* > A_row_indices(A->row_ind_dev,&A->row_ind_dev[A->nnz]);
    cusp::array1d_view< int* > A_column_indices(A->col_ind_dev,&A->col_ind_dev[A->nnz]);
    cusp::array1d_view< int* > B_column_indices(B->col_ind_dev,&B->col_ind_dev[B->nnz]);
    cusp::array1d_view< double* > A_values(A->values_dev,&A->values_dev[A->nnz]);
    cusp::array1d_view< double* > B_values(B->values_dev,&B->values_dev[B->nnz]);

    thrust::scatter_if(thrust::device,
                       thrust::make_permutation_iterator(B_row_offsets.begin(), A_column_indices.begin()) + begin_segment,
                       thrust::make_permutation_iterator(B_row_offsets.begin(), A_column_indices.begin()) + end_segment,
                       output_ptr.begin() + begin_segment,
                       segment_lengths.begin() + begin_segment,
                       B_gather_locations.begin() - output_ptr[begin_segment]); 

    thrust::inclusive_scan_by_key(thrust::device,
                                  A_gather_locations.begin(), A_gather_locations.end(),
                                  B_gather_locations.begin(),
                                  B_gather_locations.begin());

    thrust::gather(thrust::device,
                   A_gather_locations.begin(), A_gather_locations.end(),
                   A_row_indices.begin(),
                   I.begin());
    thrust::gather(thrust::device,
                   B_gather_locations.begin(), B_gather_locations.end(),
                   B_column_indices.begin(),
                   J.begin());

    thrust::transform(thrust::device,
                      thrust::make_permutation_iterator(A_values.begin(), A_gather_locations.begin()),
                      thrust::make_permutation_iterator(A_values.begin(), A_gather_locations.end()),
                      thrust::make_permutation_iterator(B_values.begin(), B_gather_locations.begin()),
                      V.begin(),
                      thrust::multiplies<double>());

    // sort (I,J,V) tuples by (I,J)
    cusp::sort_by_row_and_column(thrust::device, I, J, V);

	// compute unique number of nonzeros in the output
    IndexType NNZ = thrust::inner_product(thrust::device,
                                          thrust::make_zip_iterator(thrust::make_tuple(I.begin(), J.begin())),
                                          thrust::make_zip_iterator(thrust::make_tuple(I.end (),  J.end()))   - 1,
                                          thrust::make_zip_iterator(thrust::make_tuple(I.begin(), J.begin())) + 1,
                                          IndexType(0),
                                          thrust::plus<IndexType>(),
                                          thrust::not_equal_to< thrust::tuple<IndexType,IndexType> >()) + 1;

    // allocate space for output
	C->nnz=NNZ;
    DevMalloc((void**)&C->row_ind_dev,sizeof(int)*(C->row+1));
    DevMalloc((void**)&C->col_ind_dev,sizeof(int)*(NNZ));
    DevMalloc((void**)&C->values_dev,sizeof(double)*(NNZ));

    // sum values with the same (i,j)
    thrust::reduce_by_key
    (thrust::device,
     thrust::make_zip_iterator(thrust::make_tuple(I.begin(), J.begin())),
     thrust::make_zip_iterator(thrust::make_tuple(I.end(),   J.end())),
     V.begin(),
     thrust::make_zip_iterator(thrust::make_tuple(C->row_ind_dev, C->col_ind_dev)),
     C->values_dev,
     thrust::equal_to< thrust::tuple<IndexType,IndexType> >(),
     thrust::plus<double>());

}



double COO_MUL_COO_DEV(CooMatrixDev* A, CooMatrixDev* B, CooMatrixDev* C, anonymouslib_timer_gpu* ref_timer_gpu)
{
    ref_timer_gpu->start();

    C->row=A->row;
    C->col=B->col;


	typedef int IndexType;
    typedef double ValueType;
    typedef typename cusp::device_memory MemorySpace;

	if (A->nnz == 0 || B->nnz == 0)
    {
		C->nnz=0;
        return 0.0;
    }


    cusp::array1d_view< int* > B_row_indices(B->row_ind_dev,&B->row_ind_dev[B->nnz]);

	cusp::array1d<IndexType, MemorySpace> B_row_offsets(B->row + 1);

	cusp::indices_to_offsets(thrust::device, B_row_indices, B_row_offsets);

	cusp::array1d<IndexType, MemorySpace> B_row_lengths(B->row);

	thrust::transform(thrust::device, B_row_offsets.begin() + 1, B_row_offsets.end(), B_row_offsets.begin(), B_row_lengths.begin(), thrust::minus<IndexType>());

    cusp::array1d<IndexType, MemorySpace> segment_lengths(A->nnz);

	thrust::gather(thrust::device,
                   A->col_ind_dev, &A->col_ind_dev[A->nnz],
                   B_row_lengths.begin(),
                   segment_lengths.begin());

	cusp::array1d<IndexType, MemorySpace> output_ptr(A->nnz + 1);

	thrust::exclusive_scan(thrust::device,
                           segment_lengths.begin(), segment_lengths.end(),
                           output_ptr.begin(),
                           IndexType(0));
	
	output_ptr[A->nnz] = output_ptr[A->nnz - 1] + segment_lengths[A->nnz - 1];

	size_t coo_num_nonzeros = output_ptr[A->nnz];

    size_t workspace_capacity = thrust::min<size_t>(coo_num_nonzeros, 16 << 20);

	{
        size_t free, total;
        cudaMemGetInfo(&free, &total);

        // divide free bytes by the size of each workspace unit
        size_t max_workspace_capacity = free / (4 * sizeof(IndexType) + sizeof(ValueType));

        // use at most one third of the remaining capacity
        workspace_capacity = thrust::min<size_t>(max_workspace_capacity / 3 , workspace_capacity);
    }

	cusp::array1d<IndexType, MemorySpace> A_gather_locations;
    cusp::array1d<IndexType, MemorySpace> B_gather_locations;
    cusp::array1d<IndexType, MemorySpace> I;
    cusp::array1d<IndexType, MemorySpace> J;
    cusp::array1d<ValueType, MemorySpace> V;

/*
	if (coo_num_nonzeros <= workspace_capacity)
    {
        // compute C = A * B in one step
        size_t begin_row      = 0;
        size_t end_row        = A->row;
        size_t begin_segment  = 0;
        size_t end_segment    = A->nnz;
        size_t workspace_size = coo_num_nonzeros;

        coo_spmm_helper(workspace_size,
                        begin_row, end_row,
                        begin_segment, end_segment,
                        A, B, C,
                        B_row_offsets,
                        segment_lengths, output_ptr,
                        A_gather_locations, B_gather_locations,
                        I, J, V);
    }
    else
*/
    {

        std::list<CooMatrixDev> slices;

        // storage for C[slice,:] partial results
        //ContainerList slices;

		cusp::array1d<int, cusp::device_memory> A_row_offsets(A->row + 1);
        cusp::array1d_view< int* > A_row_ind_dev(A->row_ind_dev,&A->row_ind_dev[A->nnz]);

		cusp::indices_to_offsets(thrust::device, A_row_ind_dev, A_row_offsets);

        cusp::array1d<int, cusp::device_memory> cummulative_row_workspace(A->row);

		thrust::gather(thrust::device,
                       A_row_offsets.begin() + 1, A_row_offsets.end(),
                       output_ptr.begin(),
                       cummulative_row_workspace.begin());

        size_t begin_row = 0;
        size_t total_work = 0;

		while (begin_row < size_t(A->row))
        {

            CooMatrixDev C_slice;

            // find largest end_row such that the capacity of [begin_row, end_row) fits in the workspace_capacity
            size_t end_row = thrust::upper_bound(thrust::device,
                                                 cummulative_row_workspace.begin() + begin_row, cummulative_row_workspace.end(),
                                                 IndexType(total_work + workspace_capacity)) - cummulative_row_workspace.begin();

            size_t begin_segment = A_row_offsets[begin_row];
            size_t end_segment   = A_row_offsets[end_row];

            // TODO throw exception signaling that there is insufficient memory (not necessarily bad_alloc)
            //if (begin_row == end_row)
            //    // workspace wasn't large enough, throw cusp::memory_allocation_failure?

            size_t workspace_size = output_ptr[end_segment] - output_ptr[begin_segment];

            total_work += workspace_size;

            // TODO remove these when an exception is in place
            assert(end_row > begin_row);
            assert(workspace_size <= workspace_capacity);

            coo_spmm_helper(workspace_size, begin_row, end_row, begin_segment, end_segment, A, B, &C_slice, B_row_offsets, segment_lengths, output_ptr, A_gather_locations, B_gather_locations, I, J, V);

            C_slice.row=end_row-begin_row+1;
            C_slice.row=B->col;

            slices.push_back(C_slice);

            begin_row = end_row;
        }


		size_t C_num_entries = 0;
        for(std::list<CooMatrixDev>::iterator iter = slices.begin(); iter != slices.end(); ++iter)
            C_num_entries += iter->nnz;

		// allocate space for output
		C->row=A->row;
		C->col=B->col;
		C->nnz=C_num_entries;
    	DevMalloc((void**)&C->row_ind_dev,sizeof(int)*(C->row+1));
    	DevMalloc((void**)&C->col_ind_dev,sizeof(int)*(C_num_entries));
    	DevMalloc((void**)&C->values_dev,sizeof(double)*(C_num_entries));

        // copy slices into output
        size_t base = 0;
        cusp::array1d_view< int* > C_row_indices(C->row_ind_dev,&C->row_ind_dev[C->nnz]);
        cusp::array1d_view< int* > C_col_indices(C->col_ind_dev,&C->col_ind_dev[C->nnz]);
        cusp::array1d_view< double* > C_values_indices(C->values_dev,&C->values_dev[C->nnz]);
        for(std::list<CooMatrixDev>::iterator iter = slices.begin(); iter != slices.end(); ++iter)
        {
            cusp::array1d_view< int* > iter_row_indices(iter->row_ind_dev,&iter->row_ind_dev[iter->nnz]);
            cusp::array1d_view< int* > iter_col_indices(iter->col_ind_dev,&iter->col_ind_dev[iter->nnz]);
            cusp::array1d_view< double* > iter_values_indices(iter->values_dev,&iter->values_dev[iter->nnz]);
            thrust::copy(thrust::device, iter_row_indices.begin(),    iter_row_indices.end(),    C_row_indices.begin() + base);
            thrust::copy(thrust::device, iter_col_indices.begin(),    iter_col_indices.end(),    C_col_indices.begin() + base);
            thrust::copy(thrust::device, iter_values_indices.begin(),    iter_values_indices.end(),    C_values_indices.begin() + base);
            base += iter->nnz;
        }

    }


/* 
    B_row_ind_dev;
    DevMalloc((void**)&B_row_ind_dev,sizeof(int)*(B->row+1));

    coo_to_csr<<<1,1>>>(B->row,B->nnz,B->row_ind_dev,B_row_ind_dev);

    int* B_row_length;
    DevMalloc((void**)&B_row_length,sizeof(int)*(B->row));
    thrust::transform(thrust::device, &B_row_ind_dev[1], &B_row_ind_dev[B->row+1], &B_row_ind_dev[0], &B_row_length[0], thrust::minus<int>());

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

if (coo_num_nonzeros <= workspace_capacity)
        {

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

	thrust::scatter_if(thrust::device, thrust::make_permutation_iterator(B_row_ind_dev, A->col_ind_dev), thrust::make_permutation_iterator(B_row_ind_dev, A->col_ind_dev) + end_segment, output_ptr, segment_lengths, B_gather_locations);

    thrust::inclusive_scan_by_key(thrust::device, A_gather_locations, &A_gather_locations[workspace_size], B_gather_locations, B_gather_locations);


	thrust::gather(thrust::device,
                   A_gather_locations, &A_gather_locations[workspace_size],
                   A->row_ind_dev,
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

    }
    else
    {


		typedef typename cusp::coo_matrix<int,double,cusp::device_memory> Container;
        typedef typename std::list<Container> ContainerList;

        // storage for C[slice,:] partial results
        ContainerList slices;

		cusp::array1d<int, cusp::device_memory> A_row_offsets(A->row + 1);
        cusp::array1d_view< int* > A_row_ind_dev(A->row_ind_dev,&A->row_ind_dev[A->nnz]);

		cusp::indices_to_offsets(thrust::device, A_row_ind_dev, A_row_offsets);

        cusp::array1d<int, cusp::device_memory> cummulative_row_workspace(A->row);

		thrust::gather(thrust::device,
                       A_row_offsets.begin() + 1, A_row_offsets.end(),
                       output_ptr,
                       cummulative_row_workspace.begin());

        size_t begin_row = 0;
        size_t total_work = 0;

    }

*/

    double time_ref=ref_timer_gpu->stop();

    return time_ref;

}



__global__ void getsum_coo_cu(double* sum, int nnz, double* values)
{
for(int i=0;i<nnz;i++)
    sum[0]+=values[i];
}

double getsum_coo(CooMatrixDev *A)
{
double sum;
double *sum_dev;
DevMalloc((void**)&sum_dev,sizeof(double));
getsum_coo_cu<<<1,1>>>(sum_dev, A->nnz, A->values_dev);
DevDownload(&sum, sum_dev, sizeof(double));
cudaFree(sum_dev);
return sum;
}

double sizeofcoo(CooMatrixDev *A_coo_dev)
{
    double coo_size=0;
    coo_size+=(sizeof(int))*(A_coo_dev->row+1+A_coo_dev->nnz*2+3);
    coo_size+=(sizeof(double)*A_coo_dev->nnz);
    return coo_size;
}








void FreeCooMatrixDev(CooMatrixDev *A)
{
    cudaFree(A->row_offset_dev);
    cudaFree(A->row_ind_dev);
    cudaFree(A->col_ind_dev);
    cudaFree(A->values_dev);
    A->row=0;
    A->col=0;
    A->nnz=0;
    //free(A_csr);
    A=NULL;
}



#endif // COO_DEV_COMMON_H
