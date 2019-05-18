set(SOURCES
    svd.cpp
    svd_blocked.cpp
    svd_blocked_less_copy.cpp
    svd_blocked_less_copy_transposed.cpp
    svd_subprocedure.cpp
    svd_subprocedure_vectorized.cpp
    svd_subprocedure_vectorized_with_transpose.cpp
    svd_subprocedure_vectorized_rowwise.cpp
    nsvd.cpp
    cost.cpp
    )

add_library(two-sided-svd-lib STATIC ${SOURCES})
target_include_directories(two-sided-svd-lib PUBLIC .)
target_link_libraries(two-sided-svd-lib utils-lib)
