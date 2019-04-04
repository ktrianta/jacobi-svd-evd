#pragma once
#include <string>

/**
 * Run an external command given by cmd, and return the results outputted to STDOUT as a string.
 */
std::string exec(const char* cmd);

/**
 * Read a certain number of whitespace separated T types into the array beginning at out.
 */
template <typename T>
void read_into(std::istream& stream, T* out, size_t n) {
    for (int i = 0; i < n; ++i) stream >> out[i];
}
