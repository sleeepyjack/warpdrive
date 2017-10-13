#include <iostream>

#include "../include/pIvghor.cuh"
#include "../tools/binary_io.h"

int main(int argc, char const *argv[]) {

    constexpr uint64_t num_gpus = 2, batch_size = 1UL<<25, capacity = 1UL << 28;

    typedef pIvghor::pIvghor<num_gpus, batch_size> hashmap_t;
    typedef hashmap_t::data_t data_t;
    hashmap_t hashmap(capacity);

    TIMERSTART(create_data)
    uint64_t len_data_h = uint64_t(capacity*0.9);
    data_t * data_h = nullptr;
    cudaMallocHost(&data_h, sizeof(data_t)*len_data_h);                  CUERR
    for (uint64_t i = 0; i < len_data_h; i++) {
        data_h[i].set_key(i);
        data_h[i].set_value(i);
    }
    TIMERSTOP(create_data)

    TIMERSTART(insert)
    hashmap.insert_from_host(data_h, len_data_h);
    TIMERSTOP(insert)

    cudaFreeHost(data_h);                                                CUERR

}
