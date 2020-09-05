#ifndef CUSP_COMMON_H
#define CUSP_COMMON_H

#include "../common.h"
#include "../utils.h"
#include "../format.h"
#include "../utime.h"

#include <cusp/csr_matrix.h>
#include <cusp/multiply.h>

#define ANONYMOUSLIB_X86_CACHELINE   64


void CSR_to_CUSP(CsrMatrix* A_csr, CUSP_CSR_Host* A_csr_cusp)
{

    for(int i=0;i<=A_csr_cusp->num_rows;i++)
    {
        A_csr_cusp->row_offsets[i]=A_csr->row_ind[i];
    }
    for(int i=0;i<A_csr_cusp->num_entries;i++)
    {
        A_csr_cusp->column_indices[i]=A_csr->col_ind[i];
        A_csr_cusp->values[i]=A_csr->values[i];
    }

}




double sizeofcusp(CUSP_CSR_Host *A_cusp)
{
    double cusp_size=0;
    cusp_size+=(sizeof(int))*(A_cusp->num_rows+1+A_cusp->num_entries+3);
    cusp_size+=(sizeof(double))*(A_cusp->num_entries);
    return cusp_size;
}





#endif // CUSP_COMMON_H
