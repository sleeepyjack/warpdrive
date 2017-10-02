#pragma once

template<
    class DataPolicy = warpdrive::policies::PackedPairDataPolicy<>,
    class Index = uint64_t,
    Index GroupSize = 16,
    class FailurePolicy = warpdrive::policies::PrintIdFailurePolicy>
class BasicPlan
{

public:

    using index_t   = Index;
    using data_p    = DataPolicy;
    using failure_p = FailurePolicy;
    using data_t    = typename DataPolicy::data_t;
    using key_t     = typename DataPolicy::key_t;
    using value_t   = typename DataPolicy::value_t;
    using failure_t = typename FailurePolicy::data_t;

    static constexpr index_t group_size = GroupSize;
    static constexpr key_t   nil_key    = DataPolicy::empty_key;
    static constexpr key_t   tomb_key   = DataPolicy::tomb_key;

    enum class table_op_t
    {
        insert, retrieve, update
    };

    class config_t
    {
    public:

        config_t(index_t lvl1_max          = 100000,
                 index_t lvl2_max          = 1,
                 index_t blocks_per_grid   = std::numeric_limits<uint32_t>::max(),
                 index_t threads_per_block = 256)
                 :
                 //outer probing scheme (random hashing)
                 lvl1_max(lvl1_max),
                 //inner probing scheme (linear probing with group width)
                 lvl2_max(lvl2_max),
                 blocks_per_grid(blocks_per_grid),
                 threads_per_block(threads_per_block)
        {

        }

        index_t lvl1_max;
        index_t lvl2_max;
        index_t blocks_per_grid;
        index_t threads_per_block;
    };

    template<table_op_t TableOp,
             class      ElementOp = typename data_p::nop_op>
    HOSTQUALIFIER
    static void table_operation(data_t *     data,
                                index_t      len_data,
                                data_t *     hash_table,
                                index_t      capacity,
                                failure_p    failure_handler,
                                cudaStream_t stream = 0,
                                config_t     config = config_t())
    {

        using elem_op  = ElementOp;
        auto  table_op = TableOp;

        const auto groups_per_block = SDIV(config.threads_per_block, group_size);
        const auto blocks_needed    = SDIV(len_data, groups_per_block);

        const dim3 block_dim(group_size, groups_per_block, 1);
        const dim3 grid_dim(std::min(blocks_needed, config.blocks_per_grid), 1, 1);

        generic_kernel
        <<<grid_dim, block_dim, 0, stream>>>
        ([=] DEVICEQUALIFIER {

            const auto block = cg::this_thread_block();
            const auto group = cg::tiled_partition<GroupSize>(block);

            //grid stride loop
            for (auto data_index = get_group_id();
                      (data_index < len_data);
                      data_index += gridDim.x * blockDim.y)
            {

                auto state = state_t::neutral;
                const auto data_elem = data[data_index];

                //outer probing scheme (random hashing)
                for (auto lvl1 = 0;
                          group.all(state == state_t::neutral)
                          && (lvl1 < config.lvl1_max);
                          ++lvl1)
                {

                    auto table_index = data_p::hash(data_elem.get_key(),
                                                    lvl1);

                    //inner probing scheme (linear probing with group)
                    for (auto lvl2 = 0;
                              group.all(state == state_t::neutral)
                              && (lvl2 < config.lvl2_max);
                              ++lvl2)
                    {
                        const auto lvl2_offset = lvl2 * group.size()
                                               + group.thread_rank();

                        table_index = (table_index + lvl2_offset) % capacity;

                        auto table_elem = hash_table[table_index];

                        //update+retrieve
                        if (table_elem.get_key() == data_elem.get_key())
                        {
                            auto active = group.ballot(true);

                            //the leader
                            if (group.thread_rank() == __ffs(active)-1)
                            {
                                //update
                                elem_op::op(hash_table + table_index,
                                            data_elem);

                                //retrieve
                                if (table_op == table_op_t::retrieve)
                                {
                                    data[data_index] = table_elem;
                                }

                                //success (update or update+retrieve)
                                state = state_t::success;
                                //printf("retrieve success %d at %d\n", data_index, table_index);

                            }
                        }

                        //insert+update or failure
                        while (!group.any((state == state_t::success) || (state == state_t::failure))
                               && (table_elem.get_key() == nil_key
                               ||  table_elem.get_key() == tomb_key))
                        {

                            auto active = group.ballot(true);

                            //the leader
                            if (group.thread_rank() == __ffs(active)-1)
                            {

                                if (table_op == table_op_t::insert)
                                {
                                    //insert
                                    if (data_p::try_lockfree_swap(hash_table
                                                                  + table_index,
                                                                  table_elem,
                                                                  data_elem))
                                    {
                                        //update
                                        elem_op::op(hash_table + table_index,
                                                    data_elem);

                                        //success (insert+update)
                                        state = state_t::success;
                                        //printf("insert success %d at %d\n", data_index, table_index);
                                    }
                                }
                                else
                                {
                                    //failure (key not found)
                                    state = state_t::failure;
                                }

                                //reload candidate slots
                                table_elem = hash_table[table_index];

                            }
                        }
                    }
                }

                //if no success
                if (!group.any(state == state_t::success))
                {
                    //printf("%d %d\n", data_index, state);
                    if (group.thread_rank() == 0) {
                        //handle failure
                        failure_handler.handle(data_elem);
                    }
                }
            }
        });
    }

private:

    enum class state_t
    {
        neutral, success, failure
    };

    DEVICEQUALIFIER INLINEQUALIFIER
    static unsigned int get_group_id()
    {
        return blockDim.y * blockIdx.x + threadIdx.y;
    }

};
