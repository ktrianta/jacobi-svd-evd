#!/usr/bin/env bash

mkdir -p build
cd build
cmake "$@" ..
make
make install
ctest -V
