#ifndef BINARY_IO_H
#define BINARY_IO_H

#include <string>
#include <fstream>

using namespace std;

template <typename T>
void dump_binary(
    const T * data,
    const size_t length,
    string filename) {

    ofstream ofile(filename.c_str(), ios::binary);
    ofile.write((char*) data, sizeof(T)*length);
    ofile.close();
}

template <typename T>
void load_binary(
    const T * data,
    const size_t length,
    string filename) {

    ifstream ifile(filename.c_str(), ios::binary);
    ifile.read((char*) data, sizeof(T)*length);
    ifile.close();
}

#endif
