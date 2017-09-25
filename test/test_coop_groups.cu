#include <iostream>
#include <cstdint>
#include "cooperative_groups.h"
#include "../include/helpers/cuda_helpers.cuh"

namespace cg = cooperative_groups;

DEVICEQUALIFIER INLINEQUALIFIER
void print_bits_32(uint32_t x, uint32_t id1 = 0, uint32_t id2 = 0)
{
    unsigned char * bytes = (unsigned char*) &x;
    unsigned char   bits[32];

    for (auto i = 3; i >= 0; i--)
    {
        for (auto j = 7; j >= 0; j--) {
            unsigned char bit = (bytes[i] >> j) & 1;
            bits[i * 8 + j] = bit;
        }
    }

    printf("%u%u%u%u%u%u%u%u%u%u%u%u%u%u%u%u%u%u%u%u%u%u%u%u%u%u%u%u%u%u%u%u\n",
            bits[0], bits[1], bits[2], bits[3], bits[4], bits[5], bits[6], bits[7], bits[8], bits[9],
            bits[10], bits[11], bits[12], bits[13], bits[14], bits[15], bits[16], bits[17], bits[18], bits[19],
            bits[20], bits[21], bits[22], bits[23], bits[24], bits[25], bits[26], bits[27], bits[28], bits[29],
            bits[30], bits[31]);
}

DEVICEQUALIFIER INLINEQUALIFIER unsigned int group_mask(){
    return ((1ULL<<blockDim.x)-1)<<((blockDim.x*threadIdx.y)%32);
}

DEVICEQUALIFIER INLINEQUALIFIER
unsigned int lanemask32_lt()
{
    unsigned int lanemask32_lt;
    asm volatile("mov.u32 %0, %%lanemask_lt;" : "=r"(lanemask32_lt));
    return (lanemask32_lt);
}

DEVICEQUALIFIER INLINEQUALIFIER
unsigned int lanemask32_eq()
{
    unsigned int lanemask32_eq;
    asm volatile("mov.u32 %0, %%lanemask_eq;" : "=r"(lanemask32_eq));
    return (lanemask32_eq);
}
GLOBALQUALIFIER
void kernel(uint32_t * count)
{
    auto block = cg::this_thread_block();
    auto g     = cg::tiled_partition<16>(block);

    bool success = false;
    if (!(g.thread_rank() % 3)) {
        auto active = g.ballot(true);
        do
        {
            if (g.thread_rank() == __ffs(active)-1) //leader
            {
                printf("%u %u\n", g.thread_rank(), atomicAdd(count, 1));
                success = true;
            }
            else
            {
                active = g.ballot(true); //active but not leader
            }
        } while(active && !g.any(success));
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        printf("%u\n", count[0]);
    }
}

int main(int argc, char const *argv[]) {

    uint32_t * count; cudaMalloc(&count, sizeof(uint32_t)); CUERR
    cudaMemset(count, 0, sizeof(uint32_t)); CUERR

    kernel<<<1, dim3(16, 2, 1)>>>(count); CUERR

    cudaDeviceSynchronize(); CUERR

    cudaFree(count); CUERR

    return 0;
}
