cmake_minimum_required(VERSION 3.9.2)
set(SOURCES
    util.cpp
    util.hpp
    types.hpp
    )

add_library(utils-lib ${SOURCES})
target_include_directories(utils-lib PUBLIC .)
