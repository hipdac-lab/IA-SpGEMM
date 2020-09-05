#include <iostream>
#include <cmath>

#include "detail/dense/common_dense.h"
#include "detail/csr/common_csr.h"
#include "detail/coo/common_coo.h"
#include "detail/dia/common_dia.h"
#include "detail/ell/common_ell.h"

#include "mmio.h"
#include <pthread.h> 
#include <unistd.h>
#include <Python.h>

using namespace std;

#ifndef VALUE_TYPE
#define VALUE_TYPE double
#endif

#ifndef NUM_RUN
#define NUM_RUN 100
#endif

anonymouslib_timer ref_timer;

CsrMatrix A_csr,B_csr,C_csr;
int enable_csr=0;
double csr_time=0.0;

DiaMatrix A_dia,B_dia,C_dia;
int enable_dia=0;
double dia_time=0.0;

EllMatrix A_ell,B_ell,C_ell;
int enable_ell=0;
double ell_time=0.0;

CooMatrix A_coo,B_coo,C_coo;
int enable_coo=0;
double coo_time=0.0;

void *thread_fun_csr(void *arg)  
{
    enable_csr=0;
    pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, NULL);
    pthread_setcanceltype(PTHREAD_CANCEL_ASYNCHRONOUS, NULL);

    ref_timer.start();
    CSR_MUL_CSR(&A_csr,&B_csr,&C_csr);
    csr_time = ref_timer.stop();

    enable_csr=1;
}

void *thread_fun_dia(void *arg)  
{
    enable_dia=0;
    pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, NULL);
    pthread_setcanceltype(PTHREAD_CANCEL_ASYNCHRONOUS, NULL);

    ref_timer.start();
    DIA_mul_DIA(&A_dia,&B_dia,&C_dia);
    dia_time = ref_timer.stop();

    enable_dia=1;
}

void *thread_fun_ell(void *arg)  
{
    enable_ell=0;
    pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, NULL);
    pthread_setcanceltype(PTHREAD_CANCEL_ASYNCHRONOUS, NULL);

    ref_timer.start();
    ELL_MUL_ELL(&A_ell,&B_ell,&C_ell);
    ell_time = ref_timer.stop();

    enable_ell=1;
}

void *thread_fun_coo(void *arg)  
{
    enable_coo=0;
    pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, NULL);
    pthread_setcanceltype(PTHREAD_CANCEL_ASYNCHRONOUS, NULL);

    ref_timer.start();
    COO_MUL_COO(&A_coo,&B_coo,&C_coo);
    coo_time = ref_timer.stop();

    enable_coo=1;
}



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
    cout << "please use command like this : ./spgemm ./sample.mtx" << endl;
    exit(0);
    }
    //cout << "--------------" << filename << "--------------" << endl;
    cout << filename << ",";

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
        csrValA[i] = i%10;
    }

    //cout << " ( " << m << ", " << n << " ) nnz = " << nnzA << endl;

    A_csr.row=m;
    A_csr.col=n;
    A_csr.nnz=nnzA;
    A_csr.row_ind=csrRowPtrA;
    A_csr.col_ind=csrColIdxA;
    A_csr.values=csrValA;
    
    //printf("A_csr:\n");
    //print_csr(&A_csr);


    int job[6] = {0,0,0,0,0,1};
    int ret;
    B_csr.row=A_csr.col;
    B_csr.col=A_csr.row;
    B_csr.nnz=A_csr.nnz;
    B_csr.row_ind=(int*)malloc1d((B_csr.row+1),sizeof(int));
    B_csr.col_ind=(int*)malloc1d((B_csr.nnz),sizeof(int));
    B_csr.values=(double*)malloc1d((B_csr.nnz),sizeof(double));
    mkl_dcsrcsc (job, &A_csr.row, A_csr.values, A_csr.col_ind, A_csr.row_ind, B_csr.values, B_csr.col_ind, B_csr.row_ind, &ret);

    double ref_time;
    int time_scale=20;

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

    double transfer_formates[3];
    double run_formates[5];
    double size_formates[5];
    double sum_formates[5];
    double features[26];
    for(int i=0;i<26;i++)
        features[i]=0.0;

    GetInfo1(&A_csr,features);
    GetInfo1(&B_csr,&features[9]);

    ref_timer.start();
    CSRtoDIA(&A_csr,&A_dia);
    CSRtoDIA(&B_csr,&B_dia);
    transfer_formates[2] = ref_timer.stop();

    GetInfo2(&A_dia,&features[18]);
    GetInfo2(&B_dia,&features[21]);

    ref_timer.start();
    CSRtoELL(&A_csr,&A_ell);
    CSRtoELL(&B_csr,&B_ell);
    transfer_formates[3] = ref_timer.stop();

    ref_timer.start();
    CSRtoCOO(&A_csr,&A_coo);
    CSRtoCOO(&B_csr,&B_coo);
    transfer_formates[4] = ref_timer.stop();

    GetInfo3(&A_ell,&features[24]);
    GetInfo3(&B_ell,&features[25]);

	Py_Initialize();

    PyRun_SimpleString("import sys");  
    PyRun_SimpleString("sys.path.append('./')");  

    PyObject *pModule = PyImport_ImportModule("MatNet");
    PyObject *pDict = PyModule_GetDict(pModule);

    PyObject *pFunc = PyDict_GetItemString(pDict, "Pred");

    PyObject *pArg = Py_BuildValue("(d,d,d,d,d,d,d,d,d,d,d,d,d,d,d,d,d,d,d,d,d,d,d,d,d,d)" ,features[0],features[1],features[2],features[3],features[4],features[5],features[6],features[7],features[8],features[9],features[10],features[11],features[12],features[13],features[14],features[15],features[16],features[17],features[18],features[19],features[20],features[21],features[22],features[23],features[24],features[25]);

    PyObject *result = PyEval_CallObject(pFunc, pArg);
    int c;
    PyArg_Parse(result, "i", &c);
    printf("The Chosen One = Algorithm %d\n", c+1); 




//////////////----------MKL_START----------//////////////////////////////////////

    MKLMatrix A_mkl,B_mkl;
    A_mkl.row=A_csr.row;
    A_mkl.col=A_csr.col;
    A_mkl.nnz=A_csr.nnz;
    A_mkl.row_ind=(MKL_INT *)mkl_malloc((A_mkl.row+1)*sizeof(int),128);
    A_mkl.col_ind=(MKL_INT *)mkl_malloc((A_mkl.nnz)*sizeof(int),128);
    A_mkl.values=(VALUE_TYPE*)mkl_malloc((A_mkl.nnz)*sizeof(VALUE_TYPE),128);
    for (int i = 0; i <= A_mkl.row; i++)
    {
        A_mkl.row_ind[i]=A_csr.row_ind[i];
    }

    for (int i = 0; i < A_mkl.nnz; i++)
    {
        A_mkl.col_ind[i]=A_csr.col_ind[i];
        A_mkl.values[i]=A_csr.values[i];
    }


    B_mkl.row=B_csr.row;
    B_mkl.col=B_csr.col;
    B_mkl.nnz=B_csr.nnz;
    B_mkl.row_ind=(MKL_INT *)mkl_malloc((B_mkl.row+1)*sizeof(int),128);
    B_mkl.col_ind=(MKL_INT *)mkl_malloc((B_mkl.nnz)*sizeof(int),128);
    B_mkl.values=(VALUE_TYPE*)mkl_malloc((B_mkl.nnz)*sizeof(VALUE_TYPE),128);
    for (int i = 0; i <= B_mkl.row; i++)
    {
        B_mkl.row_ind[i]=B_csr.row_ind[i];
    }

    for (int i = 0; i < B_mkl.nnz; i++)
    {
        B_mkl.col_ind[i]=B_csr.col_ind[i];
        B_mkl.values[i]=B_csr.values[i];
    }

    MKLMatrix C_mkl;

    ref_timer.start();
    MKL_MUL_MKL(&A_mkl,&B_mkl,&C_mkl);
    ref_time = ref_timer.stop();
    run_formates[0] = ref_time;
    int mkl_time = (int)(ref_time*1000);

    double sum_mkl=0.0;
    for(int i=0;i<C_mkl.nnz;i++)
        sum_mkl+=C_mkl.values[i];

    size_formates[0]=sizeofcsr(&C_mkl);
    sum_formates[0]=sum_mkl;
   
    //print_csr(&C_mkl);

    FreeMKLMatrix(&A_mkl);
    FreeMKLMatrix(&B_mkl);
    FreeMKLMatrix(&C_mkl);

//////////////----------CSR----------//////////////////

    void *ret_pthread=NULL;  
    pthread_t tid_csr;  
    pthread_create(&tid_csr,NULL,thread_fun_csr,NULL);
    usleep(time_scale*mkl_time);  
    pthread_cancel(tid_csr);
    pthread_join(tid_csr, &ret_pthread);  
    double sum_csr=0.0;

    if(enable_csr)
    {
    for(int i=0;i<C_csr.nnz;i++)
        sum_csr+=C_csr.values[i];

    run_formates[1] = csr_time;
    size_formates[1]=sizeofcsr(&C_csr);
    sum_formates[1]=sum_csr;
    }
    else
    {
    run_formates[1] = 0.0;
    size_formates[1]= 0.0;
    sum_formates[1]=0.0;

    }

    long long flops = GetFlop(&A_csr,&B_csr);

//////////////----------DIA----------//////////////////

    ref_timer.start();
    CSRtoDIA(&A_csr,&A_dia);
    double transfer_time = ref_timer.stop();
    //print_dia(&A_dia);
    CSRtoDIA(&B_csr,&B_dia);
    //print_dia(&B_dia);
    double sum_dia=0.0;

    if(A_dia.choice && B_dia.choice)
    {
    pthread_t tid_dia;  
    pthread_create(&tid_dia,NULL,thread_fun_dia,NULL);
    usleep(time_scale*mkl_time);  
    pthread_cancel(tid_dia);
    pthread_join(tid_dia, &ret_pthread);  

    if(enable_dia)
    {
    for(int i=0;i<C_dia.row;i++)
    for(int j=0;j<C_dia.num_diagonals;j++)
        sum_dia+=C_dia.values[i][j];
    //print_dia(&C_dia);

    double diff_dia=sum_mkl-sum_dia;

    run_formates[2] = dia_time;
    size_formates[2]=sizeofdia(&C_dia);
    sum_formates[2]=sum_dia;

    }
    else
    {
    run_formates[2] = 0.0;
    size_formates[2]= 0.0;
    sum_formates[2]=0.0;
    }

    }
    else
    {
    run_formates[2] = 0.0;
    size_formates[2]= 0.0;
    sum_formates[2]=0.0;
    }
    FreeDiaMatrix(&A_dia);
    FreeDiaMatrix(&B_dia);
    FreeDiaMatrix(&C_dia);



//////////////----------ELL----------//////////////////

    ref_timer.start();
    CSRtoELL(&A_csr,&A_ell);
    transfer_time = ref_timer.stop();
    //print_ell(&A_ell);
    CSRtoELL(&B_csr,&B_ell);
    //print_ell(&B_ell);

    if(A_ell.choice && B_ell.choice)
    {
    pthread_t tid_ell;  
    pthread_create(&tid_ell,NULL,thread_fun_ell,NULL);
    usleep(time_scale*mkl_time);  
    pthread_cancel(tid_ell);
    pthread_join(tid_ell, &ret_pthread);  

    if(enable_ell)
    {
    double sum_ell=0.0;
    for(int i=0;i<C_ell.row;i++)
    for(int j=0;j<C_ell.max_nnz_per_row;j++)
        sum_ell+=C_ell.values[i][j];
    //print_ell(&C_ell);

    double diff_ell=sum_mkl-sum_ell;

    run_formates[3] = ell_time;
    size_formates[3]=sizeofell(&C_ell);
    sum_formates[3]=sum_ell;

    }
    else
    {
    run_formates[3] = 0.0;
    size_formates[3]=0.0;
    sum_formates[3]=0.0;
    }

    }
    else
    {
    run_formates[3] = 0.0;
    size_formates[3]=0.0;
    sum_formates[3]=0.0;
    }

    FreeEllMatrix(&A_ell);
    FreeEllMatrix(&B_ell);
    FreeEllMatrix(&C_ell);


//////////////----------COO----------//////////////////

    ref_timer.start();
    CSRtoCOO(&A_csr,&A_coo);
    transfer_time = ref_timer.stop();
    //print_coo(&A_coo);
    CSRtoCOO(&B_csr,&B_coo);
    //print_coo(&B_coo);
    double sum_coo=0.0;

    if(A_ell.choice && B_ell.choice)
    {
    pthread_t tid_coo;  
    pthread_create(&tid_coo,NULL,thread_fun_coo,NULL);
    usleep(time_scale*mkl_time);  
    pthread_cancel(tid_coo);
    pthread_join(tid_coo, &ret_pthread);  

    if(enable_coo)
    {
    for(int i=0;i<C_coo.nnz;i++)
        sum_coo+=C_coo.values[i];
    //print_coo(&C_coo);

    double diff_coo=sum_mkl-sum_coo;

    run_formates[4] = coo_time;
    size_formates[4]=sizeofcoo(&C_coo);
    sum_formates[4]=sum_coo;


    }
    else
    {
    run_formates[4] = 0.0;
    size_formates[4]=0.0;
    sum_formates[4]=sum_coo;
    }

    }
    else
    {
    run_formates[4] = 0.0;
    size_formates[4]=0.0;
    sum_formates[4]=sum_coo;
    }

    FreeCooMatrix(&A_coo);
    FreeCooMatrix(&B_coo);
    FreeCooMatrix(&C_coo);


    _mm_free(csrRowPtrA);
    _mm_free(csrColIdxA);
    _mm_free(csrValA);
    Py_Finalize();

    double speedup_formates[5];
    double max_speedup=0.0;
    int max_index=-1;
    for(int i=0;i<5;i++)
    {
        speedup_formates[i]=(run_formates[i]==0.0)?0.0:(run_formates[0]/run_formates[i]);
        if(max_speedup<speedup_formates[i])
        {
            max_speedup=speedup_formates[i];
            max_index=i;
        }
    }

    for(int i=0;i<5;i++)
    {
        printf("------------------------------\n");
        printf("Algorithm %d:\n",i+1);
        printf("run_time: %lf\n",run_formates[i]);
        printf("trans_time: %lf\n",transfer_formates[i]);
        printf("memory_size: %lf\n",size_formates[i]);
        printf("verified_sum: %lf\n",sum_formates[i]);
        printf("Gflops: %lf\n",(run_formates[i]==0.0)?0.0:((flops*2.0)/(run_formates[i]*1000000)));
        printf("Speedup: %lf\n",speedup_formates[i]);
    }
    printf("------------------------------\n");

    printf("MAX SPEED IS %lf for ALGORITHM %d\n", max_speedup, max_index+1);
    printf("------------------------------\n");
    if(c==max_index)
        printf("Congratulate! MatNet Correct Prediction.\n");
    else
        printf("Unfortunately! MatNet Incorrect Prediction.\n");
    printf("------------------------------\n");

    return 0;
}


