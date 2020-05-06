add_executable(
    kokkos_pi
        kokkos_pi.cpp
)
target_link_libraries(kokkos_pi PUBLIC parthenon)
