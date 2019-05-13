cmake_minimum_required(VERSION 3.12)
set(SOURCES
    util.cpp
    matrix.cpp
    block.cpp
    debug.cpp
    )

add_subdirectory(boost-cmake)

add_library(utils-lib ${SOURCES})
target_include_directories(utils-lib PUBLIC .)
message("boost source ${BOOST_SOURCE}")
target_include_directories(utils-lib PUBLIC ${BOOST_SOURCE})
