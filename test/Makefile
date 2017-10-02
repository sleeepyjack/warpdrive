NVCC  = nvcc
ARCH  = sm_61
STD   = c++11
FLAGS = -O3 -std=$(STD) -arch=$(ARCH) --expt-extended-lambda -Xcompiler -fopenmp

plan:
	$(NVCC) $(FLAGS) test_plan.cu -o test_plan

