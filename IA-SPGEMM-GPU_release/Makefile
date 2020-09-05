#compilers

#GLOBAL_PARAMETERS
VALUE_TYPE = double
NUM_RUN = 1

#ENVIRONMENT_PARAMETERS
MKLROOT = /opt/intel/parallel_studio_xe_2018/compilers_and_libraries_2018/linux/mkl
MKL_LIB = /opt/intel/parallel_studio_xe_2018/compilers_and_libraries_2018/linux/mkl/lib/intel64

#CUDA_PARAMETERS
#NVCC_FLAGS = -O3  -w -m64 -arch=sm_30
NVCC_FLAGS = -O3 -w -m64 -gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_35,code=compute_35
CUDA_INSTALL_PATH = /usr/local/cuda-8.0
CUDA_LIBS = -L$(CUDA_INSTALL_PATH)/lib64 -lcudart -lcusparse
CUDA_INCLUDES = -I$(CUDA_INSTALL_PATH)/include -I${MKLROOT}/include -L${MKLROOT}/lib/intel64 --linker-options ${MKL_LIB}/libmkl_intel_lp64.a,${MKL_LIB}/libmkl_sequential.a,${MKL_LIB}/libmkl_core.a,-lpthread

PYTHON_EVN=-I/usr/include/python3.5 -L/usr/lib/python3.5/config-3.5m-x86_64-linux-gnu -lpython3.5

cuda:
	nvcc $(NVCC_FLAGS) $(PYTHON_EVN) main.cu -o spgemm-gpu $(CUDA_INCLUDES) $(CUDA_LIBS) -D VALUE_TYPE=$(VALUE_TYPE) -D NUM_RUN=$(NUM_RUN)
