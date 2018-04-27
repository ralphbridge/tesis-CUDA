Test: kdfortran.f90 kdcuda.o
	ifort -L/usr/local/cuda-9.1/lib64 -I/usr/local/cuda-9.1/include -lcudart -lcuda -lcurand kdfortran.f90 kdcuda.o -lstdc++ -o fortCUDA
kdcuda.o: kdcuda.cu
	nvcc -c -O3 kdcuda.cu -arch=sm_61
clean:
	rm fortCUDA kdcuda.o
