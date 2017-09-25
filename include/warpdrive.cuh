#pragma once

#include <cstdint>
#include <type_traits>
#include <limits>

#include "cuda_helpers.cuh"
#include "cooperative_groups.h"

namespace cg = cooperative_groups; //FIXME namespace pollution

namespace warpdrive
{

    namespace hashers
    {

        #include "hashers.cuh"

    }; //namespace hashers

    namespace policies
    {

        #include "policies/data_policies.cuh"
        #include "policies/failure_policies.cuh"

    } //namespace policies

    namespace plans
    {

        #include "plans/basic_plan.cuh"

    }; //namespace plans

}; //namespace warpdrive
