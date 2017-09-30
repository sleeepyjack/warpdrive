#include "random_distributions.h"
#include "binary_io.h"
#include <cstdint>

int main(int argc, char const *argv[]) {
    using namespace std;
    typedef uint32_t T;
    size_t numElements = 2147483648;
    T min = numeric_limits<T>::min();
    T max = numeric_limits<T>::max()-2;
    vector<T> data = unique_random<T>(numElements, min, max);
    dump_binary<T>(data.data(), numElements, "~/Datasets/random/unique_keys_2147483648.bin");
    return 0;
}
