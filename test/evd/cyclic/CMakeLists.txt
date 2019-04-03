add_executable(evd_cyclic_test evd_cyclic_test.cpp)
add_executable(evd_cyclic_tol_test evd_cyclic_tol_test.cpp)

target_link_libraries(evd_cyclic_test gtest_main evd_cyclic-lib utils-lib m)
target_link_libraries(evd_cyclic_tol_test gtest_main evd_cyclic-lib utils-lib m)

add_test(NAME EVD_cyclic COMMAND evd_cyclic_test)
add_test(NAME EVD_cyclic_tol COMMAND evd_cyclic_tol_test)
