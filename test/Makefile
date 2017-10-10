NVCC  = nvcc
ARCH  = sm_52
STD   = c++11
FLAGS = -O3 -std=$(STD) -arch=$(ARCH) -Xcompiler -fopenmp --expt-extended-lambda

plan:
	$(NVCC) $(FLAGS) test_plan.cu -o test_plan

gossip: test_gossip.cu
	$(NVCC) $(FLAGS) test_gossip.cu -o test_gossip
