#ifndef RANDOM_DISTRIBUTION_H
#define RANDOM_DISTRIBUTION_H

#include <iostream>
#include <vector>
#include <algorithm>
#include <limits>
#include <cstddef>
#include <numeric>
#include "tbb/parallel_sort.h"
#include <boost/math/special_functions/zeta.hpp>

using namespace std;

template<typename T>
vector<T> uniform_random(size_t n, T min = numeric_limits<T>::min(), T max = numeric_limits<T>::max()){
    static_assert(std::is_fundamental<T>::value, "ERROR: Type T must be fundamental type.");
    assert(n>0 && "ERROR n must be greater than 0.");
    assert(min < max && "ERROR min must be smaller than max.");

    vector<T> data(n);
    random_device rd;
    mt19937 g(rd());
    uniform_int_distribution<T> dis(min, max);

    cout << "generating random items..." << endl;
    for (size_t i = 0; i < n; i++) {
        data[i] = dis(g);
    }
    cout << "done!" << endl << endl;
    return data;
}

template<typename T>
vector<T> unique_random(size_t n, T min = numeric_limits<T>::min(), T max = numeric_limits<T>::max()) {
    vector<T> data = uniform_random(n, min, max);

    cout << "removing duplicates iteratively..." << endl;
    random_device rd;
    mt19937 g(rd());
    uniform_int_distribution<T> dis(min, max);
    size_t delta;
    do {
        //sort(data.begin(), data.end());
        tbb::parallel_sort(data.begin(), data.end());
        data.erase(unique(data.begin(), data.end()), data.end());

        delta = n - data.size();
        cout << "\t" << float(delta)/float(n)*100 << "% duplicates..." << endl;
        for (size_t i = 0; i < delta; i++) {
            data.push_back(dis(g));
        }
    } while(delta > 0.0);
    cout << "shuffling..." << endl;
    shuffle(data.begin(), data.end(), g);
    cout << "done!" << endl << endl;
    return data;
}

template<typename T>
vector<T> zeta_distribution(size_t n, size_t k, double s, T min = numeric_limits<T>::min(), T max = numeric_limits<T>::max()) {
    assert(s>1.0 && "ERROR s must be greater than 1.0.");
    vector<T> unique = unique_random(n, min, max);
    vector<T> data;
    vector<double> pmd;
    double zeta = boost::math::zeta<double>(s);
    cout << "zeta(s=" << s << ")=" << zeta << endl;
    cout << "generating probability mass distribution..." << endl;
    for (size_t i = 1; i <= k; i++) {
        double x = pow(i, -s);
        pmd.push_back(x/zeta);
    }
    cout << "distribution:" << endl;
    for (int i=0; i<30; ++i) {
        cout << "|" << string(pmd[i]*100,'*') << pmd[i] << std::endl;
    }
    cout << "generating zeta distributed samples..." << endl;
    for (size_t i = 0; i < pmd.size(); i++) {
        size_t pn = pmd[i]*n;
        if (pn == 0){
            break;
        }
        for (size_t j = 0; j < pn; j++) {
            data.push_back(unique[i]);
        }
    }
    for (size_t i = data.size(); i < n; i++) {
        data.push_back(unique[i]);
    }
    random_device rd;
    mt19937 g(rd());
    cout << "shuffling..." << endl;
    shuffle(data.begin(), data.end(), g);
    cout << "done!" << endl << endl;
    return data;
}

#endif /* RANDOM_DISTRIBUTION_H */
