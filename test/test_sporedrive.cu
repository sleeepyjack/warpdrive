#include <iostream>

#include "../include/sporedrive.cuh"
#include "../tools/binary_io.h"

int main(int argc, char const *argv[]) {

    constexpr uint64_t num_gpus=4, num_async=1,
                       batch_size=1UL<<26, capacity=1UL<<32;

    typedef sporedrive::sporedrive<num_gpus, num_async, batch_size> hashmap_t;
    typedef hashmap_t::data_t data_t;
    hashmap_t hashmap(capacity);

    TIMERSTART(create_data)
    uint64_t len_data_h = uint64_t(capacity*0.9);
    data_t * data_h = nullptr;
    cudaMallocHost(&data_h, sizeof(data_t)*len_data_h);                  CUERR

    #pragma omp parallel for
    for (uint64_t i = 0; i < len_data_h; i++) {
        data_h[i].set_key(i);
        data_h[i].set_value(i+1);
    }
    TIMERSTOP(create_data)

    TIMERSTART(insert)
    hashmap.insert_from_host(data_h, len_data_h);
    TIMERSTOP(insert)

    //reset data
    #pragma omp parallel for
    for (uint64_t i = 0; i < len_data_h; i++) {
        data_h[i].set_value(0);
    }

    TIMERSTART(retrieve)
    hashmap.retrieve_to_host(data_h, len_data_h);
    TIMERSTOP(retrieve)

    //validate result
    uint64_t errors = 0;
    #pragma omp parallel for reduction(+:errors)
    for (uint64_t i = 0; i < len_data_h; i++) {
        if (data_h[i].get_value() != data_h[i].get_key()+1) {
            errors += 1;
        }
    }
    std::cout << "ERRORS: " << errors << std::endl;

    cudaFreeHost(data_h);                                                CUERR

}
