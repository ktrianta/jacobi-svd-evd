cmake_minimum_required(VERSION 3.9.2)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -Werror -O3 -std=c++11")

add_subdirectory(utils)
add_subdirectory(svd)
add_subdirectory(evd)
