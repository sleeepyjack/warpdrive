#include <iostream>
#include "../include/policies.cuh"

template<class DataPolicy,
         class data_t = typename DataPolicy::data_t>
GLOBALQUALIFIER
void kernel(data_t * arr, data_t value)
{
    arr[0] = data_t{0, 0};
    DataPolicy::try_insert(arr, 1, value);
}

int main(int argc, char const *argv[]) {
    using data_p = warpdrive::policies::PackedPairDataPolicy<>;
    using data_t = data_p::data_t;
    using key_t  = data_p::key_t;

    data_t d{5, 3};

    data_t * arr; cudaMalloc(&arr, sizeof(data_t));

    kernel<data_p><<<1, 1>>>(arr, d); CUERR

    cudaDeviceSynchronize(); CUERR

    std::cout << sizeof(d) << std::endl;

    return 0;
}
