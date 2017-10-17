#include <iostream>
#include "../include/gossip.cuh"
#include "../include/cuda_helpers.cuh"

template <
    uint8_t num_gpus=4,
    uint8_t max_gpus=num_gpus,
    uint8_t verbosity=2,
    typename value_t=double>
void check (
    uint64_t * device_ids_ = 0,
    uint64_t   batch_size=1UL<<4) {

    static_assert(num_gpus <= max_gpus,
                  "choose less or equal GPUs out of the existing ones.");


    uint64_t device_ids[max_gpus];
    for (uint64_t gpu = 0; gpu < max_gpus; gpu++)
        device_ids[gpu] = device_ids_ ? device_ids_[gpu] : gpu;

    // initialize RNG
    std::random_device rd;
    std::mt19937 engine(rd());

    if (!device_ids) {
        // scramble device ids using Fisher Yates
        std::uniform_int_distribution<uint64_t> tau;
        for (uint64_t src = max_gpus; src > 1; --src) {
            uint64_t trg = tau(engine) % (src-1);
            std::swap(device_ids[src-1], device_ids[trg]);
        }
    }

    // print current device ids
    if (verbosity > 0) {
        std::cout << "STATUS: device identifiers: ";
        for (uint64_t gpu = 0; gpu < num_gpus; ++gpu)
            std::cout << device_ids[gpu] << " ";
        std::cout << std::endl;
    }

    gossip::context_t<num_gpus> context(device_ids);
    gossip::memory_manager_t<num_gpus> memory_manager(&context);
    gossip::point2point_t<num_gpus> point2point(&context);
    gossip::all2all_t<num_gpus> all2all(&context);
    gossip::multisplit_t<num_gpus> multisplit(&context);
    gossip::experiment_t<num_gpus> experiment(&context);

    // determine lengths of source arrays
    value_t ** srcs, ** srcs_host,
            ** dsts, ** dsts_host;
    uint64_t srcs_lens[num_gpus], dsts_lens[num_gpus];

    // determine lengths of source and destination arrays
    for (uint64_t gpu = 0; gpu < num_gpus; ++gpu) {
        srcs_lens[gpu] = batch_size;
        dsts_lens[gpu] = batch_size;
    }

    // show srcs_lens and dsts_lens
    if (verbosity > 0) {
        std::cout << "STATUS: srcs_lens: ";
        for (uint64_t gpu = 0; gpu < num_gpus; ++gpu)
            std::cout << srcs_lens[gpu] << " ";
        std::cout << std::endl;

        std::cout << "STATUS: dsts_lens: ";
        for (uint64_t gpu = 0; gpu < num_gpus; ++gpu)
            std::cout << dsts_lens[gpu] << " ";
        std::cout << std::endl;
    }

    // create host-sided vectors for each gpu
    TIMERSTART(alloc_host_and_zero)
    srcs_host = memory_manager.template alloc_host<value_t>(srcs_lens);
    dsts_host = memory_manager.template alloc_host<value_t>(dsts_lens);
    TIMERSTOP(alloc_host_and_zero)

    TIMERSTART(alloc_device_and_zero)
    srcs = memory_manager.template alloc_device<value_t>(srcs_lens);
    dsts = memory_manager.template alloc_device<value_t>(dsts_lens);
    TIMERSTOP(alloc_device_and_zero)

    TIMERSTART(create_random_data_host)
    experiment.create_random_data_host(srcs_host, srcs_lens);
    TIMERSTOP(create_random_data_host)

    // print sources arrays
    if (verbosity > 1) {
        std::cout << "srcs[gpu] entries:" << std::endl;
        for (uint64_t gpu = 0; gpu < num_gpus; ++gpu) {
            std::cout << "gpu " << gpu << ": ";
            for (uint64_t i = 0; i < srcs_lens[gpu]; ++i)
                std::cout << srcs_host[gpu][i] << " ";
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    uint64_t table[num_gpus][num_gpus] = {0};
    const uint64_t trans_entries = num_gpus*batch_size;
    BANDWIDTHSTART(H2D)
    point2point.execH2DAsync(srcs_host, srcs, srcs_lens);
    point2point.sync();
    BANDWIDTHSTOP(H2D, trans_entries*sizeof(value_t))

    BANDWIDTHSTART(multisplit)
    multisplit.execAsync(srcs, srcs_lens, dsts, dsts_lens, table,
                         gossip::part_hash<num_gpus>());
    multisplit.sync();
    value_t ** tmp = srcs; srcs = dsts; dsts = tmp;
    BANDWIDTHSTOP(multisplit, trans_entries*sizeof(value_t))

    // print sources arrays
    if (verbosity > 1) {
        point2point.execD2HAsync(srcs, srcs_host, srcs_lens);
        point2point.sync();

        std::cout << "srcs[gpu] entries:" << std::endl;
        for (uint64_t gpu = 0; gpu < num_gpus; ++gpu) {
            std::cout << "gpu " << gpu << ": ";
            for (uint64_t i = 0; i < srcs_lens[gpu]; ++i)
                std::cout << srcs_host[gpu][i] << " ";
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    // show partition table
    if (verbosity > 0) {
        std::cout << "STATUS: partition table:" << std::endl;
        for (uint64_t gpu = 0; gpu < num_gpus; ++gpu)
            for (uint64_t part = 0; part < num_gpus; ++part)
                std::cout << (part == 0 ? "STATUS: |" : "")
                          << table[gpu][part]
                          << (part+1 == num_gpus ? "|\n" : " ");
    }

    BANDWIDTHSTART(all2all)
    uint64_t all2all_entries = 0;

    for (uint64_t gpu = 0; gpu < num_gpus; ++gpu)
        for (uint64_t part = 0; part < num_gpus; ++part)
            if (gpu != part) all2all_entries += table[gpu][part];

    if (verbosity > 0)
        all2all.print_connectivity_matrix();

    auto success = all2all.execAsync(srcs, srcs_lens, dsts, dsts_lens, table);
    all2all.sync();

    if (verbosity > 0)
        if (success)
            std::cout << "STATUS: (SUCCESS) all2all." << std::endl;
        else
            std::cout << "STATUS: (FAILURE) all2all." << std::endl;
    BANDWIDTHSTOP(all2all, all2all_entries*sizeof(value_t))

    // check dsts for correct entries
    BANDWIDTHSTART(D2H)
    point2point.execD2HAsync(dsts, dsts_host, dsts_lens);
    point2point.sync();
    BANDWIDTHSTOP(D2H, trans_entries*sizeof(value_t))

    TIMERSTART(validation)
    experiment.validate_all2all_host(dsts_host, dsts_lens, table);
    TIMERSTOP(validation)

    if (verbosity > 1) {
        for (uint64_t gpu = 0; gpu < num_gpus; ++gpu) {
            std::cout << "gpu " << gpu << ": ";
            for (uint64_t i = 0; i < dsts_lens[gpu]; ++i)
                std::cout << dsts_host[gpu][i] << " ";
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    // free memory
    memory_manager.free_host(srcs_host);
    memory_manager.free_host(dsts_host);
    memory_manager.free_device(srcs);
    memory_manager.free_device(dsts);

    std::cout << std::endl;

}

int main (int argc, char* argv[]) {


    uint64_t device_ids[2]={0, 1};
    check<2>(device_ids);

/*
    for (uint64_t gpu0=0; gpu0 < 4; ++gpu0) {
        uint64_t device_ids[1] = {gpu0};
        check<1>(device_ids);
    }


    for (uint64_t gpu0=0; gpu0 < 4; ++gpu0)
        for (uint64_t gpu1 = 0; gpu1 < 4; ++gpu1)
            if (gpu0 != gpu1) {
                uint64_t device_ids[2] = {gpu0, gpu1};
                check<2>(device_ids);
            }

    for (uint64_t gpu0=0; gpu0 < 4; ++gpu0)
        for (uint64_t gpu1 = 0; gpu1 < 4; ++gpu1)
            for (uint64_t gpu2 = 0; gpu2 < 4; ++gpu2)
                if (gpu0 != gpu1 && gpu0 != gpu2 && gpu1 != gpu2) {
                        uint64_t device_ids[3] = {gpu0, gpu1, gpu2};
                        check<3>(device_ids);
                    }

    {
        uint64_t device_ids[4] = {0, 1, 2, 3};
        check<4>(device_ids);

    }
*/
}
