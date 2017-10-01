#include <iostream>

#include "../include/warpdrive.cuh"
#include "../tools/binary_io.h"

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

    cout << "==========WARPDRIVE TEST/BENCHMARK==========" << endl;

    if (argc < 9)
    {
        cerr << "ERROR: Not enough parameters (read \"PARAMS\" section inside the main function)" << endl;
        return -1;
    }

    //the global index type to use
    using index_t = uint64_t;

    //PARAMS
    //the size of the thread groups (must be available at compile time)
    static constexpr index_t group_size = 16;
    //output verbosity (must be available at compile time)
    static constexpr index_t verbosity = 1;
    //filename for test data (dumped with binary_io.h)
    const string  filename = argv[1];
    //length of test data
    const index_t len_data = atoi(argv[2]);
    //load factor of the hash table
    const float   load     = atof(argv[3]);
    //capacity of the hash table
    const index_t capacity = len_data/load;
    //max chaotic probing attempts
    const index_t lvl1_max = atoi(argv[4]);
    //max linear probing attempts
    const index_t lvl2_max = atoi(argv[5]);
    //number of CUDA blocks per grid
    const index_t blocks_per_grid = (atoi(argv[6]) != 0) ? atoi(argv[6]): std::numeric_limits<index_t>::max();
    //number of threads per CUDA block (must be multiple of group_size)
    const index_t threads_per_block = atoi(argv[7]);
    //id of selected CUDA device
    const index_t device_id = atoi(argv[8]);

    if (verbosity > 0)
    {
        cout << "================== PARAMS =================="
             << "\n(static) group_size=" << group_size
             << "\n(static) verbosity=" << verbosity
             << "\nfilename=" << filename
             << "\nlen_data=" << len_data
             << "\nload=" << load
             << "\ncapacity=" << capacity
             << "\nlvl1_max=" << lvl1_max
             << "\nlvl2_max=" << lvl2_max
             << "\nblocks_per_grid=" << blocks_per_grid
             << "\nthreads_per_block=" << threads_per_block
             << "\ndevice_id=" << device_id
             << endl;
    }

    //DECLS
    //data policy
    using data_p    = warpdrive::policies::PackedPairDataPolicy<>;
    //failure policy
    using failure_p = std::conditional<(verbosity > 1),
                                       warpdrive::policies::PrintIdFailurePolicy,
                                       warpdrive::policies::IgnoreFailurePolicy>::type;

   //data types
   using data_t    = data_p::data_t;
   using key_t     = data_p::key_t;
   using value_t   = data_p::value_t;

    //plan (the meat)
    using plan_t = plans::BasicPlan<data_p,
                                    index_t,    //index type to use
                                    group_size, //size of parallel probing group
                                    failure_p>;

    //config struct (probing lengths and kernel launch config)
    plan_t::config_t config(lvl1_max,
                            lvl2_max,
                            blocks_per_grid,
                            threads_per_block);

    //set the selected CUDA device
    cudaSetDevice(device_id); CUERR

    //load random keys
    key_t * keys_h = (key_t*)malloc(sizeof(key_t)*len_data);
    load_binary(keys_h, len_data, filename);

    free(keys_h);
    /*
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
    plan_t::table_operation<plan_t::table_op_t::insert>(data1_d, len_data, hash_table_d, capacity, failure_handler, 0, config);
    TIMERSTOP(insert)

    cudaDeviceSynchronize(); CUERR

    failure_handler.fetch();
    failure_handler.init();


    TIMERSTART(retrieve)
    plan_t::table_operation<plan_t::table_op_t::retrieve>(data2_d, len_data, hash_table_d, capacity, failure_handler, 0, config);
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
    */
}
