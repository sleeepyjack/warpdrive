#pragma once

#include <cstdint>
#include <type_traits>
#include <limits>

#include "cuda_helpers.cuh"

namespace warpdrive
{
namespace policies
{

#include "policies/multiplicity_policies.cuh"
#include "policies/data_policies.cuh"
#include "policies/failure_policies.cuh"

}; //namespace policies
}; //namespace warpdrive
