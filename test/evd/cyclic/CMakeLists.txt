add_executable(evd_cyclic_test evd_cyclic_test.cpp ../../test_utils.cpp)
add_executable(evd_cyclic_vectorize_test evd_cyclic_vectorize_test.cpp ../../test_utils.cpp)
add_executable(evd_cyclic_blocked_test evd_cyclic_blocked_test.cpp ../../test_utils.cpp)
add_executable(evd_cyclic_tol_test evd_cyclic_tol_test.cpp ../../test_utils.cpp)

target_link_libraries(evd_cyclic_test gtest_main evd-cyclic-lib utils-lib m)
target_link_libraries(evd_cyclic_vectorize_test gtest_main evd-cyclic-lib utils-lib m)
target_link_libraries(evd_cyclic_blocked_test gtest_main evd-cyclic-lib utils-lib m)
target_link_libraries(evd_cyclic_tol_test gtest_main evd-cyclic-lib utils-lib m)

add_test(NAME EVD_cyclic COMMAND evd_cyclic_test WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME evd_cyclic_vectorize COMMAND evd_cyclic_vectorize_test WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME evd_cyclic_blocked COMMAND evd_cyclic_blocked_test WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME EVD_cyclic_tol COMMAND evd_cyclic_tol_test WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
