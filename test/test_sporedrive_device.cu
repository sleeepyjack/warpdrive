#include <iostream>

#include "../include/sporedrive.cuh"
#include "../tools/binary_io.h"

int main(int argc, char const *argv[]) {

    constexpr uint64_t num_gpus=4, num_async=1,
                       batch_size=1UL<<26, capacity=1UL<< 32;

    typedef sporedrive::sporedrive<num_gpus, num_async, batch_size> hashmap_t;
    typedef hashmap_t::data_t data_t;
    typedef hashmap_t::data_p::key_t key_t;
    hashmap_t hashmap(capacity);

    TIMERSTART(create_data)
    uint64_t len_key_h = uint64_t(capacity*0.9);
    key_t * key_h = nullptr;
    cudaMallocHost(&key_h, sizeof(key_t)*len_key_h);                     CUERR

    #pragma omp parallel for
    for (uint64_t i = 0; i < len_key_h; i++)
        key_h[i] = i;

    TIMERSTOP(create_data)

    TIMERSTART(copy_key_H2D)
    key_t * key_d[num_gpus];
    uint64_t len_key_d = len_key_h/num_gpus;
    std::cout << "overall size " << 1.0*len_key_d*num_gpus/(1<<30) << std::endl;
    for (uint64_t gpu = 0; gpu < num_gpus; ++gpu) {
        cudaSetDevice(gpu);                                             CUERR
        cudaMalloc(&key_d[gpu], sizeof(key_t)*len_key_d);               CUERR
        cudaMemcpy(key_d[gpu], key_h+gpu*len_key_d,
                   sizeof(key_t)*len_key_d, H2D);                       CUERR
    }

    TIMERSTOP(copy_key_H2D)

    for (uint64_t gpu = 0; gpu < num_gpus; ++gpu) {
        cudaSetDevice(gpu);
        cudaDeviceSynchronize();
    } CUERR


    TIMERSTART(insert)
    hashmap.insert_distribution_from_device(key_d, len_key_d);
    TIMERSTOP(insert)

    std::cout << "\n\n" << std::endl;

    TIMERSTART(retrieve)
    hashmap.retrieve_distribution_to_device(key_d, len_key_d);
    TIMERSTOP(retrieve)

    for (uint64_t gpu = 0; gpu < num_gpus; ++gpu) {
        cudaSetDevice(gpu);
        cudaDeviceSynchronize();
    } CUERR


    cudaFreeHost(key_h);                                                CUERR
    for (uint64_t gpu = 0; gpu < num_gpus; ++gpu)
        cudaFree(key_d[gpu]);                                           CUERR

}
