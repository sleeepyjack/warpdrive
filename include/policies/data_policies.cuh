#pragma once

///////////////////////////////////////////////////////////////////////////////
//// DataPolicies
///////////////////////////////////////////////////////////////////////////////

template<
    class Payload = unsigned long long int,
    class Key     = unsigned long int,
    class Value   = unsigned long int,
    class Index   = size_t,
    Index BitsKey = 32,
    Index BitsVal = 32,
    Key   EmptyKey     = std::numeric_limits<Key>::max(),
    Value TombstoneKey = std::numeric_limits<Key>::max()-1>
class PackedPairDataPolicy
{

    static_assert(BitsKey+BitsVal <= sizeof(Payload)*8,
                  "ERROR: Too many bits for chosen datatype.");
    static_assert(BitsKey > 0 && BitsVal > 0,
                  "ERROR: All bits must be greater zero.");
    static_assert(std::is_fundamental<Payload>::value,
                  "ERROR: Type Payload must be fundamental type.");
    static_assert(std::is_unsigned<Payload>::value,
                  "ERROR: Type Payload must be unsigned type.");
    static_assert(EmptyKey != TombstoneKey,
                  "ERROR: EmptyKey and TombstoneKey may not be equal.");

public:

    using key_t     = Key;
    using value_t   = Value;
    using payload_t = Payload;

    static constexpr Index   bits_key  = BitsKey;
    static constexpr Index   bits_val  = BitsVal;
    static constexpr Payload mask_key  = (1UL << bits_key)-1;
    static constexpr Payload mask_val  = (1UL << bits_val)-1;
    static constexpr Key     empty_key = EmptyKey;
    static constexpr Value   tomb_key  = TombstoneKey;

    class data_t
    {

    public:

        Payload payload;

        HOSTDEVICEQUALIFIER
        data_t(const key_t& key, const value_t& val)
        {
            set_pair(key, val);
        }

        HOSTDEVICEQUALIFIER
        data_t(const payload_t& other)
        {
            payload = other;
        }

        HOSTDEVICEQUALIFIER INLINEQUALIFIER
        void set_pair(const key_t& key, const value_t& val)
        {
            payload = key + (val << bits_key);
        }

        //FIXME HOSTDEVICEQUALIFIER throws error
        HOSTQUALIFIER INLINEQUALIFIER
        void set_pair_safe(const key_t& key, const value_t& val)
        {
            set_pair(key < mask_key ? key : mask_key-1,
                     val < mask_val ? val : mask_val);
        }

        HOSTDEVICEQUALIFIER INLINEQUALIFIER
        void set_key_safe(const key_t& key)
        {
            set_key(key < mask_key ? key : mask_key-1);
        }

        HOSTDEVICEQUALIFIER INLINEQUALIFIER
        void set_val_safe(const value_t& val)
        {
            set_val(val < mask_val ? val : mask_val);
        }

        HOSTDEVICEQUALIFIER INLINEQUALIFIER
        Index get_key() const
        {
            return payload & mask_key;
        }

        HOSTDEVICEQUALIFIER INLINEQUALIFIER
        Index get_val() const
        {
            return (payload & (mask_val << bits_val)) >> bits_key;
        }

        HOSTDEVICEQUALIFIER INLINEQUALIFIER
        bool operator==(const data_t& other) const
        {
            return payload == other.payload;
        }

        HOSTDEVICEQUALIFIER INLINEQUALIFIER
        bool operator!=(const data_t& other) const
        {
            return payload != other.payload;
        }

    };

    DEVICEQUALIFIER INLINEQUALIFIER
    static bool try_insert(data_t * address,
                           const data_t& compare,
                           const data_t& value)
    {
        const auto result = atomicCAS(static_cast<payload_t*>(address),
                                      compare.payload,
                                      value.payload);

        return (result == compare.payload);
    }

};
