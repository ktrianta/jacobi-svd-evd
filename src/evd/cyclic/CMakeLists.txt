set(SOURCES
    evd_cyclic.cpp
    evd_cyclic_oneloop.cpp
    evd_cyclic_unroll.cpp
    evd_cyclic_vectorize.cpp
    evd_cyclic_blocked.cpp
    evd_cyclic_blocked_unroll_outer.cpp
    evd_cyclic.hpp
    nevd.cpp
    )

add_library(evd-cyclic-lib STATIC ${SOURCES})
target_include_directories(evd-cyclic-lib PUBLIC .)
target_link_libraries(evd-cyclic-lib utils-lib)
