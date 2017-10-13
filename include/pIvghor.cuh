#pragma once

#include "cuda_helpers.cuh"
#include <cstdint>
#include <random>
#include <cstring>
#include <stdexcept>


namespace pIvghor {
    #include "warpdrive.cuh"
    #include "gossip.cuh"

    template <
        uint64_t num_gpus,
        uint64_t batch_size=1UL<<20,
        class WarpdriveDataPolicy=warpdrive::policies::PackedPairDataPolicy<>,
        class WarpdriveFailurePolicy=warpdrive::policies::IgnoreFailurePolicy,
        class WarpdrivePlan=warpdrive::plans::BasicPlan<16,
                                                        WarpdriveDataPolicy,
                                                        WarpdriveFailurePolicy,
                                                        uint64_t>,
        uint64_t throw_exceptions=true>
    class pIvghor {
    public:
        typedef WarpdriveFailurePolicy failure_p;
        typedef WarpdriveDataPolicy data_p;
        typedef typename WarpdriveDataPolicy::data_t data_t;
        typedef typename WarpdriveDataPolicy::key_t key_t;
        typedef typename WarpdriveDataPolicy::value_t value_t;

    private:
        // gossip primitves
        gossip::context_t<num_gpus>     * context;
        gossip::all2all_t<num_gpus>     * all2all;
        gossip::multisplit_t<num_gpus>  * multisplit;
        gossip::point2point_t<num_gpus> * point2point;

        // double buffers and hash table
        data_t * ying[num_gpus] = {nullptr};
        data_t * yang[num_gpus] = {nullptr};
        data_t * hash_table[num_gpus] = {nullptr};

        // security factor for double buffer
        const uint64_t secure_batch_size;

        // hash map configuration
        const uint64_t capacity;
        const uint64_t capacity_per_gpu;

    public:
        pIvghor (
            uint64_t   capacity_,
            uint64_t * device_ids_=0,
            double security_factor=2.0) :
                capacity(capacity_),
                capacity_per_gpu(SDIV(capacity_, num_gpus)),
                secure_batch_size(uint64_t(security_factor*batch_size))
        {

            // all operations are issued with the help of a context
            if (device_ids_)
                context = new gossip::context_t<num_gpus>(device_ids_);
            else
                context = new gossip::context_t<num_gpus>();

            // create gossip primitves
            all2all     = new gossip::all2all_t<num_gpus>(context);
            multisplit  = new gossip::multisplit_t<num_gpus>(context);
            point2point = new gossip::point2point_t<num_gpus>(context);

            // alloc memory for each hash map on each GPU and for two
            // auxiliary arrays which store to be inserted keys and values
            for (uint64_t gpu = 0; gpu < num_gpus; ++gpu) {
                cudaSetDevice(context->get_device_id(gpu));
                cudaMalloc(&ying[gpu], sizeof(data_t)*secure_batch_size);
                cudaMalloc(&yang[gpu], sizeof(data_t)*batch_size);
                cudaMalloc(&hash_table[gpu], sizeof(data_t)*capacity_per_gpu);
            } CUERR

            // reset the hash map
            reset();
        }

        void reset (value_t init_value=1)
        {
            // construct default empty key value pair
            const data_t init_data(data_p::empty_key, init_value);

            // set all entries in all hash maps to default element
            context->sync_all_streams();

            for (uint64_t gpu = 0; gpu < num_gpus; ++gpu){
                cudaSetDevice(context->get_device_id(gpu));
                memset_kernel<<<256, 1024, 0,
                                context->get_streams(gpu)[0 % num_gpus]>>>
                    (ying[gpu], secure_batch_size, init_data);
                memset_kernel<<<256, 1024, 0,
                                context->get_streams(gpu)[1 % num_gpus]>>>
                    (yang[gpu], batch_size, init_data);
                memset_kernel<<<256, 1024, 0,
                                context->get_streams(gpu)[2 % num_gpus]>>>
                    (hash_table[gpu], capacity_per_gpu, init_data);
            } CUERR

            // reset is fully blocking
            context->sync_all_streams();
        }

        ~pIvghor ()
        {
            context->sync_hard();
            for (uint64_t gpu = 0; gpu < num_gpus; ++gpu) {
                cudaSetDevice(context->get_device_id(gpu)); CUERR
                cudaFree(ying[gpu]);        CUERR
                cudaFree(yang[gpu]);        CUERR
                cudaFree(hash_table[gpu]);  CUERR
            } CUERR

            // get rid of gossip
            delete point2point;
            delete multisplit;
            delete all2all;
            delete context;
        }

        void insert_from_host(
            data_t * data_h,     // this be better pinned memory!
            uint64_t len_data_h) const
        {

            // compute number of elements to be inserted in each round
            const uint64_t batch_stride = num_gpus*batch_size;
            const uint64_t num_batches = SDIV(len_data_h, batch_stride);
            for (uint64_t batch = 0; batch < num_batches; ++batch) {
                //TIMERSTART(all2all)
                // make sure all streams are ready for action
                context->sync_all_streams();

                // the lower and upper [exclusive] position
                // to be inserted from the host array data_h
                const uint64_t lower = batch*batch_stride;
                const uint64_t upper = std::min(lower+batch_stride, len_data_h);
                const uint64_t width = upper-lower;

                // this array partitions the width many elements into
                // num_gpus many portions of approximately equal size
                data_t * srcs[num_gpus] = {nullptr};
                data_t * dsts[num_gpus] = {nullptr};
                uint64_t lens[num_gpus] = {0};
                const uint64_t sub_batch_stride = SDIV(width, num_gpus);
                for (uint64_t gpu = 0; gpu < num_gpus; ++gpu) {
                    const uint64_t sub_lower = gpu*sub_batch_stride;
                    const uint64_t sub_upper = std::min(width,
                                               sub_lower+sub_batch_stride);
                    srcs[gpu] = data_h+lower+sub_lower;
                    dsts[gpu] = ying[gpu];
                    lens[gpu] = sub_upper-sub_lower;
                }

                // move batches to buffer ying
                point2point->execH2DAsync(srcs, dsts, lens);
                point2point->sync();

                // perform multisplit on each device
                for (uint64_t gpu = 0; gpu < num_gpus; ++gpu) {
                    srcs[gpu] = ying[gpu];
                    dsts[gpu] = yang[gpu];
                }

                using hasher_t =
                    warpdrive::hashers::murmur_integer_finalizer_hash_uint32_t;

                hasher_t hasher = hasher_t();

                auto part_hash = [=] DEVICEQUALIFIER (const data_t& x){
                    return (hasher.hash(x.get_key()) % num_gpus) + 1;
                };

                uint64_t table[num_gpus][num_gpus];
                multisplit->execAsync(srcs, lens, dsts, lens, table, part_hash);
                multisplit->sync();

                // perform all to all communication
                uint64_t srcs_lens[num_gpus];
                uint64_t dsts_lens[num_gpus];
                for (uint64_t gpu = 0; gpu < num_gpus; ++gpu) {
                    srcs_lens[gpu] = batch_size;
                    dsts_lens[gpu] = secure_batch_size;
                    srcs[gpu] = yang[gpu];
                    dsts[gpu] = ying[gpu];
                }

                all2all->execAsync(srcs, srcs_lens, dsts, dsts_lens, table);
                all2all->sync();

                // compute the lens of to be hashed keys
                uint64_t v_table[num_gpus+1][num_gpus] = {0};
                for (uint64_t gpu = 0; gpu < num_gpus; ++gpu)
                    for (uint64_t part = 0; part < num_gpus; ++part)
                         v_table[gpu+1][part] =   table[gpu][part]
                                              + v_table[gpu][part];

                for (uint64_t gpu = 0; gpu < num_gpus; ++gpu)
                    lens[gpu] = v_table[num_gpus][gpu];

                //TODO: check if really necessary
                context->sync_all_streams();

                //init failure handler
                failure_p failure_handler = failure_p();
                failure_handler.init();

                //insert or update entries
                using elem_op = typename data_p::update_op;
                static constexpr auto table_op = WarpdrivePlan::table_op_t::insert;

                //TIMERSTOP(all2all)
                //TIMERSTART(insert)
                for (uint64_t gpu = 0; gpu < num_gpus; ++gpu) {
                    cudaSetDevice(context->get_device_id(gpu));

                    //execute task
                    WarpdrivePlan::template table_operation<table_op,
                                                            elem_op>
                    (dsts[gpu], lens[gpu], hash_table[gpu], capacity_per_gpu,
                     failure_handler, context->get_streams(gpu)[0]);

                } CUERR
                //TIMERSTOP(insert)

                context->sync_all_streams();
            }


        }
    };
}
