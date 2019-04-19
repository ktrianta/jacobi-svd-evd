cmake_minimum_required(VERSION 3.9.2)
add_executable(evd_classic_test evd_classic_test.cpp)
add_executable(evd_classic_tol_test evd_classic_tol_test.cpp)

target_link_libraries(evd_classic_test gtest_main evd-classic-lib utils-lib m)
target_link_libraries(evd_classic_tol_test gtest_main evd-classic-lib utils-lib m)

add_test(NAME EVD_classic COMMAND evd_classic_test)
add_test(NAME EVD_classic_tol COMMAND evd_classic_tol_test)
