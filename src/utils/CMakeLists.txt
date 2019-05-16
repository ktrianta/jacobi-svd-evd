cmake_minimum_required(VERSION 3.9.2)
set(SOURCES
    util.cpp
    matrix.cpp
    block.cpp
    debug.cpp
    )

add_library(utils-lib ${SOURCES})
target_include_directories(utils-lib PUBLIC .)
