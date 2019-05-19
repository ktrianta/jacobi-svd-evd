add_executable(evd_cyclic_test
evd_cyclic_test.cpp
evd_cyclic_vectorize_test.cpp
evd_cyclic_oneloop_vectorize_test.cpp
evd_cyclic_unroll_outer_vectorize_test.cpp
evd_cyclic_blocked_test.cpp
evd_cyclic_tol_test.cpp
evd_cyclic_oneloop_row_test.cpp
evd_cyclic_unroll_test.cpp
evd_cyclic_oneloop_test.cpp
../../test_utils.cpp)

target_link_libraries(evd_cyclic_test gtest_main evd-cyclic-lib utils-lib m)

add_test(NAME EVD_cyclic COMMAND evd_cyclic_test WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
