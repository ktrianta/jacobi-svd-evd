set(SOURCES
    evd_cyclic.cpp
<<<<<<< HEAD
    evd_cyclic_unroll.cpp
=======
    evd_cyclic_vectorize.cpp
>>>>>>> eec9bbeac0c0a2d510c426760e330e27861df132
    evd_cyclic.hpp
    )

add_library(evd-cyclic-lib STATIC ${SOURCES})
target_include_directories(evd-cyclic-lib PUBLIC .)
target_link_libraries(evd-cyclic-lib utils-lib)
