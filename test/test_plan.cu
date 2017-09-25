#include <iostream>

#include "../include/warpdrive.cuh"

DEVICEQUALIFIER INLINEQUALIFIER
void print_bits_32(uint32_t x)
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

int main(int argc, char const *argv[]) {

    using namespace warpdrive;
    using namespace std;

    // type decls
    using plan_t  = plans::BasicPlan<>;

    using data_p     = plan_t::data_p;
    using failure_p  = plan_t::failure_p;

    using index_t = plan_t::index_t;
    using data_t  = data_p::data_t;
    using key_t   = data_p::key_t;

    // params
    const index_t len_data = atoi(argv[1]);
    const float   load     = atof(argv[2]);
    const index_t capacity = len_data/load;

    // generate data
    data_t * data1_h = new data_t[len_data];
    data_t * data2_h = new data_t[len_data];
    data_t * data1_d; cudaMalloc(&data1_d, sizeof(data_t)*len_data); CUERR
    data_t * data2_d; cudaMalloc(&data2_d, sizeof(data_t)*len_data); CUERR
    for (uint32_t i = 0; i < len_data; ++i) {
        data1_h[i] = data_t{i, i+1};
        data2_h[i] = data_t{i, plan_t::init_val};
    }
    cudaMemcpy(data1_d, data1_h, sizeof(data_t)*len_data, H2D); CUERR
    cudaMemcpy(data2_d, data2_h, sizeof(data_t)*len_data, H2D); CUERR

    data_t * hash_table_h = new data_t[capacity];
    data_t * hash_table_d; cudaMalloc(&hash_table_d, sizeof(data_t)*capacity); CUERR
    memset_kernel<<<SDIV(capacity, 1024), 1024>>>(hash_table_d, capacity, data_t{plan_t::nil_key, plan_t::init_val});

    failure_p failure_handler = failure_p();

    cudaDeviceSynchronize(); CUERR

    failure_handler.init();

    TIMERSTART(insert)
    plan_t::table_operation<plan_t::table_op_t::insert>(data1_d, len_data, hash_table_d, capacity, failure_handler);
    TIMERSTOP(insert)

    cudaDeviceSynchronize(); CUERR

    failure_handler.fetch();
    failure_handler.init();


    TIMERSTART(retrieve)
    plan_t::table_operation<plan_t::table_op_t::retrieve>(data2_d, len_data, hash_table_d, capacity, failure_handler);
    TIMERSTOP(retrieve)

    cudaDeviceSynchronize(); CUERR

    cudaMemcpy(data2_h, data2_d, sizeof(data_t)*len_data, D2H); CUERR

    cudaMemcpy(hash_table_h, hash_table_d, sizeof(data_t)*capacity, D2H); CUERR

    index_t num_errors = 0;
    for (auto i = 0; i < len_data; i++) {
        if (data1_h[i] != data2_h[i]) {
            std::cout << "Error at " << i
            << ": (" << data1_h[i].get_key() << ", " << data1_h[i].get_value()
            << ") != (" << data2_h[i].get_key() << ", " << data2_h[i].get_value()
            << ")" << std::endl;
            num_errors++;
        }
    }
    std::cout << "num_errors = " << num_errors << std::endl;

    // free memory
    delete[] data1_h;
    delete[] data2_h;
    delete[] hash_table_h;
    cudaFree(data1_d); CUERR
    cudaFree(data2_d); CUERR
    cudaFree(hash_table_d); CUERR
}
