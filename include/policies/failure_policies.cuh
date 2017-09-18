#pragma once

///////////////////////////////////////////////////////////////////////////////
//// FaultPolicies
///////////////////////////////////////////////////////////////////////////////

class IgnoreFailurePolicy
{

public:

    using return_t = int;

    HOSTQUALIFIER INLINEQUALIFIER
    explicit IgnoreFailurePolicy()
    {

    }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    ~IgnoreFailurePolicy()
    {

    }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    void init() const
    {

    }

    template<class Failure>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    void handle(const Failure& msg) const
    {

    }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    return_t fetch(const cudaStream_t& stream = 0) const
    {
        return 0;
    }

};

class PrintIdFailurePolicy
{

public:

    using return_t = int;

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    explicit PrintIdFailurePolicy()
    {

    }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    ~PrintIdFailurePolicy()
    {

    }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    void init() const
    {

    }

    template<class Failure>
    DEVICEQUALIFIER INLINEQUALIFIER
    void handle(const Failure& msg) const
    {
        printf("Fault at block (%u, %u, %u) in thread (%u, %u, %u)\n",
            blockIdx.x, blockIdx.y, blockIdx.y,
            threadIdx.x, threadIdx.y, threadIdx.z);
    }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    return_t fetch(const cudaStream_t& stream = 0) const
    {
        return 0;
    }

};

class BooleanFailurePolicy
{

public:

    using return_t = bool;

    HOSTQUALIFIER INLINEQUALIFIER
    explicit BooleanFailurePolicy()
    {
        cudaMalloc(&failure_happened, sizeof(return_t)); CUERR
    }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    ~BooleanFailurePolicy()
    {
        #ifndef __CUDACC__
        cudaFree(failure_happened); CUERR
        #endif
    }

    HOSTQUALIFIER INLINEQUALIFIER
    void init(const cudaStream_t& stream = 0) const
    {
        memset_kernel<<<1, 1, 0, stream>>>
        (failure_happened, sizeof(return_t), return_t(0)); CUERR
    }

    template<class Failure>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    void handle(const Failure& msg)
    {
        if(!(*failure_happened))
            *failure_happened = true; //NOTE race condition but correct
    }

    HOSTQUALIFIER INLINEQUALIFIER
    return_t fetch(const cudaStream_t& stream = 0) const
    {
        return_t ret;
        cudaMemcpyAsync(
            &ret, failure_happened, sizeof(return_t), D2H, stream); CUERR
        return ret;
    }

private:

    return_t * failure_happened = nullptr;

};

template<class T = unsigned int>
class CountFailurePolicy
{

public:

    using return_t = T;

    HOSTQUALIFIER INLINEQUALIFIER
    explicit CountFailurePolicy()
    {
        cudaMalloc(&count, sizeof(return_t)); CUERR
    }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    ~CountFailurePolicy()
    {
        #ifndef __CUDACC__
        cudaFree(count); CUERR
        #endif
    }

    HOSTQUALIFIER INLINEQUALIFIER
    void init(const cudaStream_t& stream = 0) const
    {
        memset_kernel<<<1, 1, 0, stream>>>
        (count, sizeof(return_t), return_t(0)); CUERR
    }

    template<class Failure>
    DEVICEQUALIFIER INLINEQUALIFIER
    void handle(const Failure& msg)
    {
        atomicAdd(count, 1);
    }

    HOSTQUALIFIER INLINEQUALIFIER
    return_t fetch(const cudaStream_t& stream = 0) const
    {
        return_t ret;
        cudaMemcpyAsync(&ret, count, sizeof(return_t), D2H, stream); CUERR
        return ret;
    }

private:

    return_t * count = nullptr;

};
