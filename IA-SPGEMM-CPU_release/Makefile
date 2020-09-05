#compilers
CC=icc

#GLOBAL_PARAMETERS
VALUE_TYPE = double
NUM_RUN = 1000

#ENVIRONMENT_PARAMETERS
PYTHON_EVN=-I/usr/include/python3.6 -L/usr/lib/python3.6/config-3.6m-x86_64-linux-gnu -lpython3.6

#backup
spgemm_cpu:
	$(CC) -g -std=c++11 $(PYTHON_EVN) -qopt-prefetch=3 -fopenmp -mkl -O3 main.cpp -o spgemm-cpu -D VALUE_TYPE=$(VALUE_TYPE) -D NUM_RUN=$(NUM_RUN)
