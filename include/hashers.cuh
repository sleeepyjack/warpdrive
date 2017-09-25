#pragma once

///////////////////////////////////////////////////////////////////////////////
// uint32_t hashers
///////////////////////////////////////////////////////////////////////////////

class nvidia_hash_uint32_t
{

public:

    using data_t = uint32_t;

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static data_t hash(data_t x)
    {

        x = (x + 0x7ed55d16) + (x << 12);
        x = (x ^ 0xc761c23c) ^ (x >> 19);
        x = (x + 0x165667b1) + (x <<  5);
        x = (x + 0xd3a2646c) ^ (x <<  9);
        x = (x + 0xfd7046c5) + (x <<  3);
        x = (x ^ 0xb55a4f09) ^ (x >> 16);

        return x;
    }
};

class mueller_hash_uint32_t
{

public:

    using data_t = uint32_t;

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static data_t hash(data_t x)
    {

        x = ((x >> 16) ^ x) * 0x45d9f3b;
        x = ((x >> 16) ^ x) * 0x45d9f3b;
        x = ((x >> 16) ^ x);

        return x;
    }
};

class murmur_integer_finalizer_hash_uint32_t
{

public:

    using data_t = uint32_t;

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static data_t hash(data_t x)
    {

        x ^= x >> 16;
        x *= 0x85ebca6b;
        x ^= x >> 13;
        x *= 0xc2b2ae35;
        x ^= x >> 16;

        return x;
    }
};

class identity_uint32_t
{

public:

    using data_t = uint32_t;

    template<class T>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static data_t hash(T x)
    {
        return data_t{x};
    }
};

///////////////////////////////////////////////////////////////////////////////
// uint64_t hashers
///////////////////////////////////////////////////////////////////////////////

class murmur_hash_3_uint64_t
{

public:

    using data_t = uint64_t;

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static data_t hash(data_t x)
    {

        x ^= x >> 33;
        x *= 0xff51afd7ed558ccd;
        x ^= x >> 33;
        x *= 0xc4ceb9fe1a85ec53;
        x ^= x >> 33;

        return x;
    }
};
