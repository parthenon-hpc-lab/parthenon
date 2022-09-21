#=========================================================================================
# (C) (or copyright) 2020. Triad National Security, LLC. All rights reserved.
#
# This program was produced under U.S. Government contract 89233218CNA000001 for Los
# Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
# for the U.S. Department of Energy/National Nuclear Security Administration. All rights
# in the program are reserved by Triad National Security, LLC, and the U.S. Department
# of Energy/National Nuclear Security Administration. The Government is granted for
# itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
# license in this material to reproduce, prepare derivative works, distribute copies to
# the public, perform publicly and display publicly, and to permit others to do so.
#=========================================================================================

add_subdirectory(stochastic_subgrid)
add_subdirectory(advection)
add_subdirectory(calculate_pi)
add_subdirectory(face_fields)
add_subdirectory(kokkos_pi)
add_subdirectory(particles)
add_subdirectory(particle_leapfrog)
add_subdirectory(particle_tracers)
add_subdirectory(poisson)
add_subdirectory(sparse_advection)
add_subdirectory(poisson_cg)
add_subdirectory(poisson_nonlinear)

