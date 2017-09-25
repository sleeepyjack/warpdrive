#include <iostream>
#include "../include/policies.cuh"

template<class FailurePolicy>
GLOBALQUALIFIER
void kernel(FailurePolicy failure_handler)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid % 2)
        failure_handler.handle(tid);
}

int main(int argc, char const *argv[]) {
    using failure_t = warpdrive::policies::BooleanFailurePolicy;

    cudaStream_t s;
    cudaStreamCreate(&s);

    failure_t failure_handler;

    failure_handler.init(s);

    kernel<<<1, 10, 0, s>>>(failure_handler);

    std:: cout << failure_handler.fetch(s) << std::endl;

    cudaStreamDestroy(s);

    return 0;
}
