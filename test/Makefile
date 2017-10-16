NVCC  = /project/pararch/cuda/cuda-9.0/bin/nvcc
ARCH  = sm_60
STD   = c++11
FLAGS = -O3 -g -lineinfo -std=$(STD) -arch=$(ARCH) -Xcompiler -fopenmp --expt-extended-lambda -Xcompiler="-Wall"

plan:   test_plan.cu
	$(NVCC) $(FLAGS) test_plan.cu -o test_plan

gossip: test_gossip.cu
	$(NVCC) $(FLAGS) test_gossip.cu -o test_gossip

pIvghor:   test_pIvghor.cu
	$(NVCC) $(FLAGS) test_pIvghor.cu -o test_pIvghor
