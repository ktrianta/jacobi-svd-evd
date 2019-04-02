set(SOURCES
    evd.cpp
    evd.hpp
    )

add_library(evd-lib STATIC ${SOURCES})
target_include_directories(evd-lib PUBLIC .)
target_link_libraries(evd-lib utils-lib)

add_executable(evd main.cpp)
target_link_libraries(evd utils-lib m)
