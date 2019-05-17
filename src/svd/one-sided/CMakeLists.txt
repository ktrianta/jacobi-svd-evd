cmake_minimum_required(VERSION 3.12)

set(SOURCES
    svd.cpp
    svd.hpp
    )

add_library(svd-lib STATIC ${SOURCES})
target_include_directories(svd-lib PUBLIC .)
target_link_libraries(svd-lib utils-lib)

add_executable(svd main.cpp)
target_link_libraries(svd utils-lib m)
