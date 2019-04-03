add_executable(evd_test evd_test.cpp)
add_executable(evd_tol_test evd_tol_test.cpp)
add_executable(evd_cyclic_test evd_cyclic_test.cpp)

target_link_libraries(evd_test gtest_main evd-lib utils-lib m)
target_link_libraries(evd_tol_test gtest_main evd-lib utils-lib m)
target_link_libraries(evd_cyclic_test gtest_main evd-lib utils-lib m)

add_test(NAME EVD COMMAND evd_test)
add_test(NAME EVD_tol COMMAND evd_tol_test)
add_test(NAME EVD_cyclic COMMAND evd_cyclic_test)
