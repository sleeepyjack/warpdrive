#pragma once

#include <string>
#include <fstream>

template <typename T>
void dump_binary(const T * data,
                 const size_t length,
                 const std::string& filename)
{
    std::ofstream ofile(filename, std::ios::binary);
    ofile.write((char*) data, sizeof(T)*length);
    ofile.close();
}

template <typename T>
void load_binary(const T * data,
                 const size_t length,
                 const std::string& filename)
{
    std::ifstream ifile(filename, std::ios::binary);
    ifile.read((char*) data, sizeof(T)*length);
    ifile.close();
}
