#include <iostream>

#include "../include/warpdrive.cuh"
#include "../tools/binary_io.h"

int main(int argc, char const *argv[]) {

    using namespace warpdrive;
    using namespace std;

    if (argc < 9)
    {
        cerr << "ERROR: Not enough parameters (read \"PARAMS\" section inside the main function)" << endl;
        return -1;
    }

    //the global index type to use
    using index_t = uint64_t;

    //PARAMS
    //the size of the thread groups (must be available at compile time)
    static constexpr index_t group_size = 4;
    //output verbosity (must be available at compile time)
    static constexpr index_t verbosity = 0;
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
        cout << "================= WARPDRIVE ================" << endl;
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
    using failure_p = std::conditional<(verbosity > 1), //if verbosity < 2 ignore failures
                                       warpdrive::policies::PrintIdFailurePolicy,
                                       warpdrive::policies::IgnoreFailurePolicy>::type;

   //data types
   using data_t  = data_p::data_t;
   using key_t   = data_p::key_t;
   using value_t = data_p::value_t;

    //plan (the meat)
    using plan_t = plans::BasicPlan<data_p,
                                    group_size, //size of parallel probing group
                                    index_t,    //index type to use
                                    failure_p>;

    //config struct (probing lengths and kernel launch config)
    plan_t::config_t config(lvl1_max,
                            lvl2_max,
                            blocks_per_grid,
                            threads_per_block);

    //set the selected CUDA device
    cudaSetDevice(device_id); CUERR

    //load random keys
    key_t * keys_h = new key_t[len_data];
    load_binary<key_t>(keys_h, len_data, filename);

    //the hash table
    data_t * hash_table_d; cudaMalloc(&hash_table_d, sizeof(data_t)*capacity); CUERR

    //test data
    data_t * data_h = new data_t[len_data];
    data_t * data_d; cudaMalloc(&data_d, sizeof(data_t)*len_data); CUERR

    //TESTS/BENCHMARKS
    if (verbosity > 0)
    {
        cout << "============= TESTS/BENCHMARK =============" << endl;
    }

    //init failure handler
    failure_p failure_handler = failure_p();
    failure_handler.init();

    //the first task to execute
    using elem_op_1 = data_p::count_op;
    static constexpr auto table_op_1 = plan_t::table_op_t::insert;

    //the second task to execute
    using elem_op_2 = data_p::nop_op;
    static constexpr auto table_op_2 = plan_t::table_op_t::retrieve;

    //init hash table
    memset_kernel
    <<<SDIV(capacity, 1024), 1024>>>
    (hash_table_d, capacity, data_t(data_p::empty_key, elem_op_1::identity));

    //init input data
    #pragma omp parallel for
    for (index_t i = 0; i < len_data; i++)
    {
        data_h[i] = data_t(keys_h[i], 0);
    }
    memcpy(data_h+(len_data/2), data_h, sizeof(data_t)*len_data/2);
    cudaMemcpy(data_d, data_h, sizeof(data_t)*len_data, H2D); CUERR

    //execute task
    TIMERSTART(op1)
    plan_t::table_operation<table_op_1,
                            elem_op_1>
    (data_d, len_data, hash_table_d, capacity, failure_handler, 0, config);
    TIMERSTOP(op1)

    //re-init data
    #pragma omp parallel for
    for (index_t i = 0; i < len_data; i++)
    {
        data_h[i].set_value(elem_op_2::identity);
    }

    cudaMemcpy(data_d, data_h, sizeof(data_t)*len_data, H2D); CUERR

    //retrieve results
    TIMERSTART(op2)
    plan_t::table_operation<table_op_2,
                            elem_op_2>
    (data_d, len_data, hash_table_d, capacity, failure_handler, 0, config);
    TIMERSTOP(op2)

    cudaMemcpy(data_h, data_d, sizeof(data_t)*len_data, D2H); CUERR

    //validation
    index_t num_errors = 0;
    //#pragma omp parallel for reduction(+:num_errors)
    for (index_t i = 0; i < len_data; i++) {
        cout << data_h[i].get_key() << "\t" << data_h[i].get_value() << endl;
        if (data_h[i].get_value() != i+1)
        {
            num_errors += 1;
        }
    }

    cout << "ERRORS: " << num_errors << endl;

    //free memory
    delete[] keys_h;
    delete[] data_h;

    cudaFree(hash_table_d);
    cudaFree(data_d);
}
