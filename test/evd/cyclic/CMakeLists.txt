add_executable(evd_cyclic_tol_test evd_cyclic_tol_test.cpp)

target_link_libraries(evd_cyclic_tol_test gtest_main evd_cyclic-lib utils-lib m)

add_test(NAME EVD_cyclic_tol COMMAND evd_cyclic_tol_test)
