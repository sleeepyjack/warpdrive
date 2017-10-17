NVCC  = /project/pararch/cuda/cuda-9.0/bin/nvcc
ARCH  = sm_60
STD   = c++11
FLAGS = -O3 -g -lineinfo -std=$(STD) -arch=$(ARCH) --expt-extended-lambda -Xcompiler="-fopenmp"

plan:   test_plan.cu
	$(NVCC) $(FLAGS) test_plan.cu -o test_plan

gossip: test_gossip.cu
	$(NVCC) $(FLAGS) test_gossip.cu -o test_gossip

pIvghor:   test_pIvghor.cu
	$(NVCC) $(FLAGS) test_pIvghor.cu -o test_pIvghor

sporedrive:   test_sporedrive.cu
	$(NVCC) $(FLAGS) test_sporedrive.cu -o test_sporedrive

sporedrive_device:   test_sporedrive_device.cu
	$(NVCC) $(FLAGS) test_sporedrive_device.cu -o test_sporedrive_device
