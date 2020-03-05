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

cmake_minimum_required(VERSION 3.10)
project(parthenon LANGUAGES C CXX)

# Options
include(CTest)
option(ENABLE_UNIT_TESTS "Enable unit tests" ${BUILD_TESTING})
option(ENABLE_INTEGRATION_TESTS "Enable integration tests" ${BUILD_TESTING})
option(ENABLE_REGRESSION_TESTS "Enable regression tests" ${BUILD_TESTING})
option(DISABLE_MPI "MPI is enabled by default if found, set this to True to disable MPI" OFF)
option(DISABLE_OPENMP "OpenMP is enabled by default if found, set this to True to disable OpenMP" OFF)
option(DISABLE_HDF5 "HDF5 is enabled by default if found, set this to True to disable HDF5" OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Don't allow in-source builds
file(TO_CMAKE_PATH "${PROJECT_BINARY_DIR}/CMakeLists.txt" LOC_PATH)
if(EXISTS "${LOC_PATH}")
    message(FATAL_ERROR
      "You cannot build in a source directory (or any directory with a CMakeLists.txt file). "
      "Please make a build subdirectory. Feel free to remove CMakeCache.txt and CMakeFiles.")
endif()

# If the user doesn't specify a build type, prefer RelWithDebInfo
set(default_build_type "RelWithDebInfo")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
      STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(ENABLE_MPI OFF)
if (NOT DISABLE_MPI)
  find_package(MPI COMPONENTS CXX)
  if (NOT MPI_FOUND)
    message(FATAL_ERROR "MPI is required but couldn't be found. "
    "If you want to build Parthenon without MPI, please rerun CMake with -DDISABLE_MPI=ON")
  endif()
  set(ENABLE_MPI ON)
endif()

set(ENABLE_OPENMP OFF)
if (NOT DISABLE_OPENMP)
  find_package(OpenMP COMPONENTS CXX)
  if (NOT OpenMP_FOUND)
    message(FATAL_ERROR "OpenMP is required but couldn't be found. "
    "If you want to build Parthenon without OpenMP, please rerun CMake with -DDISABLE_OPENMP=ON")
  endif()
  set(ENABLE_OPENMP ON)
endif()

set(ENABLE_HDF5 OFF)
if (NOT DISABLE_HDF5)
  set(HDF5_PREFER_PARALLEL ${ENABLE_MPI})
  find_package(HDF5 COMPONENTS C)
  if (NOT HDF5_FOUND)
    message(FATAL_ERROR "HDF5 is required but couldn't be found. "
    "If you want to build Parthenon without HDF5, please rerun CMake with -DDISABLE_HDF5=ON")
  endif()
  set(ENABLE_HDF5 ON)

  if (ENABLE_MPI AND (NOT HDF5_IS_PARALLEL))
    message(FATAL_ERROR "Both MPI and HDF5 are enabled but only a serial version of HDF5 "
    "was found. Please install a parallel version of HDF5 (or point CMake to it by adding its path "
    "to the CMAKE_PREFIX_PATH environment variable), or disable either MPI or HDF5 by rerunning "
    "CMake with -DDISABLE_MPI=ON or -DDISABLE_HDF5=ON")
  endif()

  # HDF5 Interface library
  add_library(HDF5_C INTERFACE)
  target_link_libraries(HDF5_C INTERFACE ${HDF5_C_LIBRARIES})
  target_compile_definitions(HDF5_C INTERFACE ${HDF5_C_DEFINITIONS})
  target_include_directories(HDF5_C INTERFACE ${HDF5_C_INCLUDE_DIRS})
endif()

# Kokkos recommendatation resulting in not using default GNU extensions
set(CMAKE_CXX_EXTENSIONS OFF)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/kokkos kokkos)

# Build Tests and download Catch2
if (${ENABLE_UNIT_TESTS} OR ${ENABLE_INTEGRATION_TESTS} OR ${ENABLE_REGRESSION_TESTS})
  # Try finding an installed Catch2 first
  find_package(Catch2 2.11.1)

  if (NOT CATCH2_FOUND)
    # If Catch2 is not found, instead use the git submodule
    if (NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/external/Catch2/single_include)
      # Unable to find the header files for Catch2 or they don't exist
      message(STATUS "Downloading Catch2 submodule.")

      # Clone the submodule
      execute_process(COMMAND git submodule update --init --force -- external/Catch2 WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    endif()

    add_subdirectory(external/Catch2)
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/external/Catch2/contrib")
  endif()

  include(ParseAndAddCatchTests)
  add_subdirectory(tst)
endif()

add_subdirectory(src)
add_subdirectory(example/calculate_pi)

include(cmake/CheckCopyright.cmake)
