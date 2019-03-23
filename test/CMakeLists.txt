cmake_minimum_required(VERSION 3.9.2)

set(SOURCES
    ../src/util.cpp
    ../src/svd.cpp
    ../src/evd.cpp
    )
set(CMAKE_CXX_FLAGS "-Wall -Wextra -O3 -std=c++11")

add_library(sources_lib STATIC ${SOURCES})

add_executable(svd_test svd_test.cpp)
add_executable(util_test util_test.cpp)
add_executable(evd_test evd_test.cpp)
target_link_libraries(svd_test gtest_main sources_lib m)
target_link_libraries(util_test gtest_main sources_lib m)
target_link_libraries(evd_test gtest_main sources_lib m)


add_test(NAME SVD COMMAND svd_test)
add_test(NAME Utilities COMMAND util_test)
add_test(NAME EVD COMMAND evd_test)
