#pragma once

template<
    uint8_t GroupSize = 16,
    class DataPolicy = warpdrive::policies::PackedPairDataPolicy<>,
    class FailurePolicy = warpdrive::policies::PrintIdFailurePolicy,
    class Index = uint64_t>
class BasicPlan
{

    using index_t   = Index;
    using data_p    = DataPolicy;
    using failure_p = FailurePolicy;
    using data_t    = typename DataPolicy::data_t;
    using key_t     = typename DataPolicy::key_t;
    using value_t   = typename DataPolicy::value_t;
    using failure_t = typename FailurePolicy::data_t;

    static constexpr index_t group_size = GroupSize;
    static constexpr key_t   empty_key  = DataPolicy::empty_key;
    static constexpr key_t   tomb_key   = DataPolicy::tomb_key;

public:

    enum class table_op_t
    {
        insert, retrieve, update
    };

    class config_t
    {
    public:

        config_t(index_t lvl1_max          = 100000,
                 index_t lvl2_max          = 1,
                 index_t blocks_per_grid   = (1UL << 31)-1,
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

            namespace cg = cooperative_groups;

            const auto block = cg::this_thread_block();
            const auto group = cg::tiled_partition<GroupSize>(block);

            state_t thread_state;
            bool group_is_active;
            data_t table_elem, data_elem;

            auto any_thread_state_changed = [&]
            {
                return !group.all(thread_state == state_t::neutral);
            };

            auto leader_rank = [&]
            {
                return __ffs(group.ballot(true))-1;
            };

            auto any_candidates = [&]
            {
                return group.any(table_elem.get_key() == data_elem.get_key()
                              || table_elem.get_key() == empty_key
                              || table_elem.get_key() == tomb_key);
            };

            //grid stride loop
            for (index_t data_index = get_group_id();
                         data_index < len_data;
                         data_index += gridDim.x * blockDim.y)
            {

                thread_state = state_t::neutral;
                group_is_active = true;
                data_elem = data[data_index];

                //outer probing scheme (random hashing)
                for (index_t lvl1 = 0;
                             group_is_active && (lvl1 < config.lvl1_max);
                             ++lvl1)
                {

                    index_t table_index = data_p::hash(data_elem.get_key(), lvl1);

                    //inner probing scheme (linear probing with group)
                    for (index_t lvl2 = 0;
                                 group_is_active && (lvl2 < config.lvl2_max);
                                 ++lvl2)
                    {

                        index_t lvl2_offset = lvl2 * group.size()
                                            + group.thread_rank();

                        table_index = (table_index + lvl2_offset) % capacity;

                        table_elem = hash_table[table_index];

                        //any candidate slots inside window?
                        while (any_candidates())
                        {

                            //update+retrieve
                            if (table_elem.get_key() == data_elem.get_key())
                            {

                                if(group.thread_rank() == leader_rank())
                                {

                                    //update
                                    elem_op::op(hash_table + table_index,
                                                data_elem);

                                    //retrieve
                                    if (table_op == table_op_t::retrieve)
                                    {
                                        data[data_index] = table_elem;
                                    }

                                    thread_state = state_t::success;
                                }
                            }

                            //sync and check group state
                            if (any_thread_state_changed())
                            {
                                group_is_active = false;
                                break;
                            }

                            //insert
                            if (table_elem.get_key() == empty_key
                             || table_elem.get_key() == tomb_key)
                            {
                                if (table_op == table_op_t::insert)
                                {
                                    if(group.thread_rank() == leader_rank())
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
                                            thread_state = state_t::success;
                                        }
                                    }
                                }
                                else
                                {
                                    if (table_elem.get_key() == empty_key)
                                    {
                                        //failure (key not found)
                                        thread_state = state_t::failure;
                                    }

                                }
                            }

                            //sync and check group state
                            if (any_thread_state_changed())
                            {
                                group_is_active = false;
                                break;
                            }

                            //reload window
                            table_elem = hash_table[table_index];
                        }
                    }
                }

                //if no success
                if (!group.any(thread_state == state_t::success))
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
