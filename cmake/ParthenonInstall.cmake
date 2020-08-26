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

target_include_directories(parthenon PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include/parthenon>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include/parthenon/generated>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/parthenon>
  )

install(TARGETS parthenon EXPORT parthenonTargets
  INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    )
#install(TARGETS parthenon EXPORT parthenonTargets
#  INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/parthenon"
#  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
#  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
#    )


# Maintain directory structure in installed include files
#install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/parthenon DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/parthenon" FILES_MATCHING PATTERN "*.hpp")
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/parthenon DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}" FILES_MATCHING PATTERN "*.hpp")

# Install generated config header file
#install(FILES ${PROJECT_SOURCE_DIR}/include/parthenon/config.hpp
#  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/parthenon")
install(FILES ${PROJECT_SOURCE_DIR}/include/parthenon/config.hpp
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

install(FILES ${PROJECT_BINARY_DIR}/cmake/parthenonConfig.cmake DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/parthenon")

install(EXPORT parthenonTargets
    FILE parthenonTargets.cmake
    NAMESPACE Parthenon::
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/parthenon"
    )
