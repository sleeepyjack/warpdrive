#pragma once

///////////////////////////////////////////////////////////////////////////////
//// DataPolicies
///////////////////////////////////////////////////////////////////////////////

template<
    class Payload      = unsigned long long int,
    class Key          = uint32_t,
    class Value        = uint32_t,
    uint8_t BitsKey    = 32,
    uint8_t BitsVal    = 32,
    Key   EmptyKey     = std::numeric_limits<Key>::max(),
    Value TombstoneKey = std::numeric_limits<Key>::max()-1,
    Value InitValue    = 0>
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

    static constexpr uint8_t   bits_key  = BitsKey;
    static constexpr uint8_t   bits_val  = BitsVal;
    static constexpr payload_t mask_key  = (1UL << bits_key)-1;
    static constexpr payload_t mask_val  = (1UL << bits_val)-1;
    static constexpr key_t     empty_key = EmptyKey;
    static constexpr value_t   tomb_key  = TombstoneKey;
    static constexpr value_t   init_val  = InitValue;

    class data_t
    {

    public:

        Payload payload;

        HOSTDEVICEQUALIFIER
        data_t()
        {

        }

        HOSTDEVICEQUALIFIER
        data_t(const key_t& key, const value_t& val)
        {
            set_pair(key, val); //TODO set_pair_safe
        }

        HOSTDEVICEQUALIFIER
        data_t(const payload_t& other)
        {
            payload = other;
        }

        HOSTDEVICEQUALIFIER INLINEQUALIFIER
        void set_pair(const key_t& key, const value_t& val)
        {
            payload = payload_t(key) + (payload_t(val) << bits_key);
        }

        //FIXME HOSTDEVICEQUALIFIER throws error
        HOSTQUALIFIER INLINEQUALIFIER
        void set_pair_safe(const key_t& key, const value_t& val)
        {
            set_pair(key < mask_key ? key : mask_key-1,
                     val < mask_val ? val : mask_val);
        }

        HOSTDEVICEQUALIFIER INLINEQUALIFIER
        void set_key(const key_t& key)
        {
            payload = (payload & ~mask_key) + key;
        }

        HOSTDEVICEQUALIFIER INLINEQUALIFIER
        void set_key_safe(const key_t& key)
        {
            set_key(payload_t(key) < mask_key ? payload_t(key) : mask_key-1);
        }

        HOSTDEVICEQUALIFIER INLINEQUALIFIER
        void set_value(const value_t& val)
        {
            payload = (payload & ~(mask_val << bits_key)) + (payload_t(val) << bits_key);
        }

        HOSTDEVICEQUALIFIER INLINEQUALIFIER
        void set_value_safe(const value_t& val)
        {
            set_value(payload_t(val) < mask_val ? payload_t(val) : mask_val);
        }


        HOSTDEVICEQUALIFIER INLINEQUALIFIER
        key_t get_key() const
        {
            return payload & mask_key;
        }

        HOSTDEVICEQUALIFIER INLINEQUALIFIER
        value_t get_value() const
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
    static bool try_lockfree_swap(data_t * address,
                                  const data_t& compare,
                                  const data_t& value)
    {
        const auto result = atomicCAS((payload_t *) address,
                                      compare.payload,
                                      value.payload);

        return (result == compare.payload);
    }

    template<class Hasher = warpdrive::hashers::mueller_hash_uint32_t>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static auto hash(const key_t& key, const key_t& i = 0)
    {
        return Hasher::hash(key + i);
    }

    class nop_op
    {
    public:

        DEVICEQUALIFIER INLINEQUALIFIER
        static void op(data_t * address, const data_t& value)
        {

        }
    };

    class update_op
    {
    public:

        DEVICEQUALIFIER INLINEQUALIFIER
        static void op(data_t * address, const data_t& value)
        {
            atomicExch((payload_t *) address, value.payload);
        }
    };


    class delete_op
    {
    public:

        DEVICEQUALIFIER INLINEQUALIFIER
        static void op(data_t * address, const data_t& value)
        {
            atomicExch((payload_t *) address,
                       data_t(TombstoneKey, InitValue).payload);
        }
    };

    class count_op
    {
    public:

        DEVICEQUALIFIER INLINEQUALIFIER
        static void op(data_t * address, const data_t& value)
        {
            data_t _old, _new;
            value_t _new_value;

            do {
                _old = _new = *address;
                _new_value = _new.get_value();
                _new.set_value_safe(++_new_value);
            } while(!try_lockfree_swap(address, _old, _new));
        }
    };

    template<value_t Max = std::numeric_limits<value_t>::max()>
    class count_clamped_op
    {
    public:

        DEVICEQUALIFIER INLINEQUALIFIER
        static void op(data_t * address, const data_t& value)
        {

            if ((*address).get_value() != Max)
            {
                count_op::op(address, value);
            }
        }
    };

};
