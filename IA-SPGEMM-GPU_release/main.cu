#include <iostream>
#include <cmath>

#include "detail/csr/common_csr.h"
#include "detail/cusp/common_cusp.h"
#include "detail/cusparse/common_cusparse.h"
#include "detail/csr_dev/common_csr_dev.h"
#include "detail/dia/common_dia.h"
#include "detail/dia_dev/common_dia_dev.h"
#include "detail/ell/common_ell.h"
#include "detail/ell_dev/common_ell_dev.h"
#include "detail/coo/common_coo.h"
#include "detail/coo_dev/common_coo_dev.h"

#include "mmio.h"
#include "mkl.h"
#include <Python.h>

using namespace std;

#ifndef VALUE_TYPE
#define VALUE_TYPE double
#endif

#ifndef NUM_RUN
#define NUM_RUN 100
#endif


int main(int argc, char ** argv)
{
    // report precision of floating-point
    //cout << "------------------------------------------------------" << endl;
    char  *precision;
    if (sizeof(VALUE_TYPE) == 4)
    {
        precision = "32-bit Single Precision";
    }
    else if (sizeof(VALUE_TYPE) == 8)
    {
        precision = "64-bit Double Precision";
    }
    else
    {
        cout << "Wrong precision. Program exit!" << endl;
        return 0;
    }

    //cout << "PRECISION = " << precision << endl;
    //cout << "------------------------------------------------------" << endl;

    int m, n, nnzA;
    int *csrRowPtrA;
    int *csrColIdxA;
    VALUE_TYPE *csrValA;

    anonymouslib_timer_cpu ref_timer_cpu;
    anonymouslib_timer_gpu ref_timer_gpu;
    ref_timer_gpu.create();

    //ex: ./spmv webbase-1M.mtx
    int argi = 1;

    char  *filename;
    if(argc > argi)
    {
        filename = argv[argi];
        argi++;
    }
    else
    {
    cout << "please use command like this : ./spgemm_csr ./sample.mtx" << endl;
    exit(0);
    }

    //cout << filename << ",";

    // read matrix from mtx file
    //printf("read matrix from mtx file.\n");
    int ret_code;
    MM_typecode matcode;
    FILE *f;

    int nnzA_mtx_report;
    int isInteger = 0, isReal = 0, isPattern = 0, isSymmetric = 0;

    // load matrix
    if ((f = fopen(filename, "r")) == NULL)
        return -1;

    if (mm_read_banner(f, &matcode) != 0)
    {
        cout << "Could not process Matrix Market banner." << endl;
        return -2;
    }

    if ( mm_is_complex( matcode ) )
    {
        cout <<"Sorry, data type 'COMPLEX' is not supported. " << endl;
        return -3;
    }

    if ( mm_is_pattern( matcode ) )  { isPattern = 1; /*cout << "type = Pattern" << endl;*/ }
    if ( mm_is_real ( matcode) )     { isReal = 1; /*cout << "type = real" << endl;*/ }
    if ( mm_is_integer ( matcode ) ) { isInteger = 1; /*cout << "type = integer" << endl;*/ }

    /* find out size of sparse matrix .... */
    ret_code = mm_read_mtx_crd_size(f, &m, &n, &nnzA_mtx_report);
    if (ret_code != 0)
        return -4;

    if ( mm_is_symmetric( matcode ) || mm_is_hermitian( matcode ) )
    {
        isSymmetric = 1;
        //cout << "symmetric = true" << endl;
    }
    else
    {
        //cout << "symmetric = false" << endl;
    }

    int *csrRowPtrA_counter = (int *)malloc((m+1) * sizeof(int));
    memset(csrRowPtrA_counter, 0, (m+1) * sizeof(int));

    int *csrRowIdxA_tmp = (int *)malloc(nnzA_mtx_report * sizeof(int));
    int *csrColIdxA_tmp = (int *)malloc(nnzA_mtx_report * sizeof(int));
    VALUE_TYPE *csrValA_tmp    = (VALUE_TYPE *)malloc(nnzA_mtx_report * sizeof(VALUE_TYPE));

    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

    for (int i = 0; i < nnzA_mtx_report; i++)
    {
        int idxi, idxj;
        double fval;
        int ival;

        if (isReal)
            fscanf(f, "%d %d %lg\n", &idxi, &idxj, &fval);
        else if (isInteger)
        {
            fscanf(f, "%d %d %d\n", &idxi, &idxj, &ival);
            fval = ival;
        }
        else if (isPattern)
        {
            fscanf(f, "%d %d\n", &idxi, &idxj);
            fval = 1.0;
        }

        // adjust from 1-based to 0-based
        idxi--;
        idxj--;

        csrRowPtrA_counter[idxi]++;
        csrRowIdxA_tmp[i] = idxi;
        csrColIdxA_tmp[i] = idxj;
        csrValA_tmp[i] = fval;
    }

    if (f != stdin)
        fclose(f);

    if (isSymmetric)
    {
        for (int i = 0; i < nnzA_mtx_report; i++)
        {
            if (csrRowIdxA_tmp[i] != csrColIdxA_tmp[i])
                csrRowPtrA_counter[csrColIdxA_tmp[i]]++;
        }
    }

    // exclusive scan for csrRowPtrA_counter
    int old_val, new_val;

    old_val = csrRowPtrA_counter[0];
    csrRowPtrA_counter[0] = 0;
    for (int i = 1; i <= m; i++)
    {
        new_val = csrRowPtrA_counter[i];
        csrRowPtrA_counter[i] = old_val + csrRowPtrA_counter[i-1];
        old_val = new_val;
    }

    nnzA = csrRowPtrA_counter[m];
    csrRowPtrA = (int *)_mm_malloc((m+1) * sizeof(int), ANONYMOUSLIB_X86_CACHELINE);
    memcpy(csrRowPtrA, csrRowPtrA_counter, (m+1) * sizeof(int));
    memset(csrRowPtrA_counter, 0, (m+1) * sizeof(int));

    csrColIdxA = (int *)_mm_malloc(nnzA * sizeof(int), ANONYMOUSLIB_X86_CACHELINE);
    csrValA    = (VALUE_TYPE *)_mm_malloc(nnzA * sizeof(VALUE_TYPE), ANONYMOUSLIB_X86_CACHELINE);

    if (isSymmetric)
    {
        for (int i = 0; i < nnzA_mtx_report; i++)
        {
            if (csrRowIdxA_tmp[i] != csrColIdxA_tmp[i])
            {
                int offset = csrRowPtrA[csrRowIdxA_tmp[i]] + csrRowPtrA_counter[csrRowIdxA_tmp[i]];
                csrColIdxA[offset] = csrColIdxA_tmp[i];
                csrValA[offset] = csrValA_tmp[i];
                csrRowPtrA_counter[csrRowIdxA_tmp[i]]++;

                offset = csrRowPtrA[csrColIdxA_tmp[i]] + csrRowPtrA_counter[csrColIdxA_tmp[i]];
                csrColIdxA[offset] = csrRowIdxA_tmp[i];
                csrValA[offset] = csrValA_tmp[i];
                csrRowPtrA_counter[csrColIdxA_tmp[i]]++;
            }
            else
            {
                int offset = csrRowPtrA[csrRowIdxA_tmp[i]] + csrRowPtrA_counter[csrRowIdxA_tmp[i]];
                csrColIdxA[offset] = csrColIdxA_tmp[i];
                csrValA[offset] = csrValA_tmp[i];
                csrRowPtrA_counter[csrRowIdxA_tmp[i]]++;
            }
        }
    }
    else
    {
        for (int i = 0; i < nnzA_mtx_report; i++)
        {
            int offset = csrRowPtrA[csrRowIdxA_tmp[i]] + csrRowPtrA_counter[csrRowIdxA_tmp[i]];
            csrColIdxA[offset] = csrColIdxA_tmp[i];
            csrValA[offset] = csrValA_tmp[i];
            csrRowPtrA_counter[csrRowIdxA_tmp[i]]++;
        }
    }

    // free tmp space
    free(csrColIdxA_tmp);
    free(csrValA_tmp);
    free(csrRowIdxA_tmp);
    free(csrRowPtrA_counter);

    srand(time(NULL));

    // set csrValA to 1, easy for checking floating-point results
    for (int i = 0; i < nnzA; i++)
    {
        //csrValA[i] = i+1.0;
        csrValA[i] = rand() % 10;
    }

    //cout << " ( " << m << ", " << n << " ) nnz = " << nnzA << endl;

    //printf("read matrix end.\n");

    CsrMatrix A_csr;
    A_csr.row=m;
    A_csr.col=n;
    A_csr.nnz=nnzA;
    A_csr.row_ind=csrRowPtrA;
    A_csr.col_ind=csrColIdxA;
    A_csr.values=csrValA;
    
    //printf("A_csr:\n");
    //print_csr(&A_csr);

    CsrMatrix B_csr;
    int job[6] = {0,0,0,0,0,1};
    int ret;
    B_csr.row=A_csr.col;
    B_csr.col=A_csr.row;
    B_csr.nnz=A_csr.nnz;
    B_csr.row_ind=(int*)malloc1d((B_csr.row+1),sizeof(int));
    B_csr.col_ind=(int*)malloc1d((B_csr.nnz),sizeof(int));
    B_csr.values=(double*)malloc1d((B_csr.nnz),sizeof(double));
    mkl_dcsrcsc (job, &A_csr.row, A_csr.values, A_csr.col_ind, A_csr.row_ind, B_csr.values, B_csr.col_ind, B_csr.row_ind, &ret);




   
//////////////----------density_representation_start----------//////////////////////////////////////



    long long** images=(long long**)malloc2d(128,128,sizeof(long long));

    for(int i=0;i<128;i++)
        for(int j=0;j<128;j++)
        {
            images[i][j]=0;
        }

    for(int i=0;i<A_csr.row;i++)
    {
        for(int j=A_csr.row_ind[i];j<A_csr.row_ind[i+1];j++)
        {
            int old_i=i;
            int old_j=A_csr.col_ind[j];
            int new_i_s=0,new_i_e=0;
            int new_j_s=0,new_j_e=0;
            if(A_csr.row>128)
            {
                new_i_s=(int)(old_i*128/A_csr.row);
                new_i_e=(int)(old_i*128/A_csr.row);
            }
            else if(A_csr.row<128)
            {
                new_i_s=(int)(old_i*128/A_csr.row);
                new_i_e=(int)(old_i*128/A_csr.row)+(int)(128/A_csr.row);
            }
            else
            {
                new_i_s=old_i;
                new_i_e=old_i;
            }

            if(A_csr.col>128)
            {
                new_j_s=(int)(old_j*128/A_csr.col);
                new_j_e=(int)(old_j*128/A_csr.col);
            }
            else if(A_csr.col<128)
            {
                new_j_s=(int)(old_j*128/A_csr.col);
                new_j_e=(int)(old_j*128/A_csr.col)+(int)(128/A_csr.col);
            }
            else
            {
                new_j_s=old_j;
                new_j_e=old_j;
            }

            for(int k=new_i_s;k<=new_i_e;k++)
                for(int m=new_j_s;m<=new_j_e;m++)
                    if(k<128&&m<128)
                        images[k][m]++;
        }
    }


    FILE *fpWrite=fopen("./imgs/img1.txt","w");

    for(int i=0;i<128;i++)
    {
        for(int j=0;j<128;j++)
        {
            fprintf(fpWrite,"%lld\n",images[i][j]);
        }
    }

    fclose(fpWrite);


    for(int i=0;i<128;i++)
        for(int j=0;j<128;j++)
        {
            images[i][j]=0;
        }

    for(int i=0;i<B_csr.row;i++)
    {
        for(int j=B_csr.row_ind[i];j<B_csr.row_ind[i+1];j++)
        {
            int old_i=i;
            int old_j=B_csr.col_ind[j];
            int new_i_s=0,new_i_e=0;
            int new_j_s=0,new_j_e=0;
            if(B_csr.row>128)
            {
                new_i_s=(int)(old_i*128/B_csr.row);
                new_i_e=(int)(old_i*128/B_csr.row);
            }
            else if(B_csr.row<128)
            {
                new_i_s=(int)(old_i*128/B_csr.row);
                new_i_e=(int)(old_i*128/B_csr.row)+(int)(128/B_csr.row);
            }
            else
            {
                new_i_s=old_i;
                new_i_e=old_i;
            }

            if(B_csr.col>128)
            {
                new_j_s=(int)(old_j*128/B_csr.col);
                new_j_e=(int)(old_j*128/B_csr.col);
            }
            else if(B_csr.col<128)
            {
                new_j_s=(int)(old_j*128/B_csr.col);
                new_j_e=(int)(old_j*128/B_csr.col)+(int)(128/B_csr.col);
            }
            else
            {
                new_j_s=old_j;
                new_j_e=old_j;
            }

            for(int k=new_i_s;k<=new_i_e;k++)
                for(int m=new_j_s;m<=new_j_e;m++)
                    if(k<128&&m<128)
                        images[k][m]++;
        }
    }

    fpWrite=fopen("./imgs/img2.txt","w");

    for(int i=0;i<128;i++)
    {
        for(int j=0;j<128;j++)
        {
            fprintf(fpWrite,"%lld\n",images[i][j]);
        }
    }

    free2d(images);

    fclose(fpWrite);

 


    CsrMatrix C_csr;

    ref_timer_cpu.start();
    CSR_MUL_CSR(&A_csr,&B_csr,&C_csr);
    double ref_time = ref_timer_cpu.stop();

    double sum_csr=0.0;
    for(int i=0;i<C_csr.nnz;i++)
        sum_csr+=C_csr.values[i];
 
    //printf("C_csr:\n");
    //print_csr(&C_csr);

    //cout << ref_time << ","<< sum_csr << ","<< sizeofcsr(&C_csr) << ",";


    double features[18];
    for(int i=0;i<18;i++)
        features[i]=0.0;

	double transfer_formates[1];
    double run_formates[2];
    double size_formates[2];
    double sum_formates[2];

    GetInfo1(&A_csr,features);
    GetInfo1(&B_csr,&features[9]);

	Py_Initialize();

    PyRun_SimpleString("import sys");  
    PyRun_SimpleString("sys.path.append('./')");  

    PyObject *pModule = PyImport_ImportModule("MatNet");
    PyObject *pDict = PyModule_GetDict(pModule);

    PyObject *pFunc = PyDict_GetItemString(pDict, "Pred");

    PyObject *pArg = Py_BuildValue("(d,d,d,d,d,d,d,d,d,d,d,d,d,d,d,d,d,d)" ,features[0],features[1],features[2],features[3],features[4],features[5],features[6],features[7],features[8],features[9],features[10],features[11],features[12],features[13],features[14],features[15],features[16],features[17]);
    PyObject *result = PyEval_CallObject(pFunc, pArg);
    int c;
    PyArg_Parse(result, "i", &c);
    printf("The Chosen One = Algorithm %d\n", c+1);


    double flops = GetFlop(&A_csr,&B_csr);



    ///////////////////////////////////CUSP_START////////////////////////////////////////

	// initialize matrix
    CUSP_CSR_Host A_csr_cusp(A_csr.row,A_csr.col,A_csr.nnz),B_csr_cusp(B_csr.row,B_csr.col,B_csr.nnz);
    CUSP_COO_Device A_coo_cusp,B_coo_cusp,C_coo_cusp;
    
    CSR_to_CUSP(&A_csr,&A_csr_cusp);
    CSR_to_CUSP(&B_csr,&B_csr_cusp);

    A_coo_cusp=A_csr_cusp;
    B_coo_cusp=B_csr_cusp;


    ref_timer_gpu.start();
    // compute y = A * x
    cusp::multiply(A_coo_cusp,B_coo_cusp,C_coo_cusp);
    ref_time = ref_timer_gpu.stop();

    CUSP_CSR_Host C_csr_cusp=C_coo_cusp;

    A_coo_cusp.resize(0, 0, 0);
    B_coo_cusp.resize(0, 0, 0);
    C_coo_cusp.resize(0, 0, 0);

    // print y
    //cusp::print(C_coo_cusp);

    double sum_cusp=0.0;
    for(int i=0;i<C_csr_cusp.num_entries;i++)
        sum_cusp+=C_csr_cusp.values[i];


    double diff_cusp=sum_csr-sum_cusp;

    run_formates[0]=ref_time;
    sum_formates[0]=sum_cusp;
    size_formates[0]=sizeofcusp(&C_csr_cusp);

    ///////////////////////////////////CUSP_END////////////////////////////////////////


    ///////////////////////////////////cuSPARSE_START//////////////////////////////////

    cuSparseMatrix A_cusparse,B_cusparse,C_cusparse;

    CSR_to_CUSPARSE(&A_csr, &A_cusparse);
    CSR_to_CUSPARSE(&B_csr, &B_cusparse);

    ref_time = CUSPARSE_MUL_CUSPARSE(&A_cusparse,&B_cusparse,&C_cusparse,&ref_timer_gpu);
    
    double diff_cusparse=sum_csr-getsum_cusparse(&C_cusparse);

    run_formates[1]=ref_time;
    sum_formates[1]=getsum_cusparse(&C_cusparse);
    size_formates[1]=sizeofcusparse(&C_cusparse);

    ///////////////////////////////////cuSPARSE_END////////////////////////////////////


    ///////call_NSPARSE///////

	for(int i=0;i<2;i++)
    {   
        printf("------------------------------\n");
        printf("Algorithm %d:\n",i+1);
        printf("run_time: %lf\n",run_formates[i]);
        printf("memory_size: %lf\n",size_formates[i]);
        printf("verified_sum: %lf\n",sum_formates[i]);
        printf("Gflops: %lf\n",(run_formates[i]==0.0)?0.0:((flops*2.0)/(run_formates[i]*1000000)));
    }
    printf("------------------------------\n");

    if(c==0)
        printf("MatNet predicts Algorithm CUSP is optimal\n");
    if(c==1)
        printf("MatNet predicts Algorithm cuSPARSE is optimal\n");
    if(c==2)
        printf("MatNet predicts Algorithm NSPARSE is optimal\n");


    FreecuSparseMatrix(&A_cusparse);
    FreecuSparseMatrix(&B_cusparse);
    FreecuSparseMatrix(&C_cusparse);


    _mm_free(csrRowPtrA);
    _mm_free(csrColIdxA);
    _mm_free(csrValA);

    return 0;
}

