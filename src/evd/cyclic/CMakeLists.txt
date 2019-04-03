set(SOURCES
    evd_cyclic.cpp
    evd_cyclic.hpp
    )

add_library(evd_cyclic-lib STATIC ${SOURCES})
target_include_directories(evd_cyclic-lib PUBLIC .)
target_link_libraries(evd_cyclic-lib utils-lib)

add_executable(evd_cyclic main.cpp)
target_link_libraries(evd_cyclic utils-lib m)
