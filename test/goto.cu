#include <iostream>
#include "../include/cuda_helpers.cuh"

GLOBALQUALIFIER
void kernel()
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (!(tid % 2)) {
        if (!(tid % 4)) {
            goto label;
        }
    }
    return;

    label:
    printf("%d at label\n", tid);
}

int main(int argc, char const *argv[]) {
    kernel<<<10, 1>>>();

    cudaDeviceSynchronize(); CUERR
    return 0;
}
