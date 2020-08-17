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

cmake_minimum_required(VERSION 3.12)
project(parthenon VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)

# Compile time constants
set(PARTHENON_NGHOST 2 CACHE STRING "Number of ghost cells") 

# Compile Options
option(PARTHENON_SINGLE_PRECISION "Run in single precision" OFF) 
option(PARTHENON_DISABLE_MPI "MPI is enabled by default if found, set this to True to disable MPI" OFF)
option(PARTHENON_DISABLE_OPENMP "OpenMP is enabled by default if found, set this to True to disable OpenMP" OFF)
option(PARTHENON_DISABLE_HDF5 "HDF5 is enabled by default if found, set this to True to disable HDF5" OFF)
option(
  PARTHENON_LINT_DEFAULT
  "Sources files are linted when building the default target. Set to OFF to \
stop linting by default. You can still lint using the \"lint\" target"
  ON
)
option(
  PARTHENON_DISABLE_EXAMPLES 
  "Example drivers are built by default, set this to True to diable, note if regression tests are \
built some, drivers needed by the regression tests will still be built." 
  OFF
)

option(PARTHENON_ENABLE_UNIT_TESTS "Enable unit tests" ${BUILD_TESTING})
option(PARTHENON_ENABLE_INTEGRATION_TESTS "Enable integration tests" ${BUILD_TESTING})
option(PARTHENON_ENABLE_PERFORMANCE_TESTS "Enable performance tests" ${BUILD_TESTING})
option(PARTHENON_ENABLE_REGRESSION_TESTS "Enable regression tests" ${BUILD_TESTING})
option(ENABLE_COMPILER_WARNINGS "Enable compiler warnings" OFF)
option(CHECK_REGISTRY_PRESSURE "Check the registry pressure for Kokkos CUDA kernels" OFF)
option(TEST_INTEL_OPTIMIZATION "Test intel optimization and vectorization" OFF)
option(TEST_ERROR_CHECKING "Enables the error checking unit test. This test will FAIL" OFF)
option(CODE_COVERAGE "Enable code coverage reporting" OFF)

include(cmake/Format.cmake)
include(cmake/Lint.cmake)

# internal variable for number of ghost cells
set(NUMBER_GHOST_CELLS ${PARTHENON_NGHOST}) 

# regression test reference data
set(REGRESSION_GOLD_STANDARD_VER 2 CACHE STRING "Version of gold standard to download and use")
set(REGRESSION_GOLD_STANDARD_HASH "SHA512=45f57d16b76a3a44940e40ea642c1d5a2b3aea681a3064e4c06b031bf264d627f8009e967b92a2cb0fbc08e5cd0ffda381038483f390fa00f47c42a551ca3646" CACHE STRING "Hash of default gold standard file to download")
option(REGRESSION_GOLD_STANDARD_SYNC "Automatically sync gold standard files." ON)

# set single precision #define
if ( PARTHENON_SINGLE_PRECISION )
  set(SINGLE_PRECISION_ENABLED 1)
else()
  set(SINGLE_PRECISION_ENABLED 0)
endif()  

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(cmake/CodeCov.cmake)
if(CODE_COVERAGE)
  if( NOT COVERAGE_NAME)
    SET(COVERAGE_NAME "coverage_reports")
  endif()
  if( NOT COVERAGE_PATH)
    SET(COVERAGE_PATH "${CMAKE_BINARY_DIR}/coverage")
  endif()
  SET(GCC_COVERAGE_COMPILE_FLAGS "--coverage")
  SET(GCC_COVERAGE_LINK_FLAGS    "--coverage")
endif()

SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}")
SET(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} ${GCC_COVERAGE_LINK_FLAGS}")
 
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
set(NUM_MPI_PROC_TESTING "4" CACHE STRING "Number of mpi processors to use when running tests with MPI")
if (NOT PARTHENON_DISABLE_MPI)
  find_package(MPI COMPONENTS CXX)
  if (NOT MPI_FOUND)
    message(FATAL_ERROR "MPI is required but couldn't be found. "
    "If you want to build Parthenon without MPI, please rerun CMake with -DPARTHENON_DISABLE_MPI=ON")
  endif()
  set(ENABLE_MPI ON)
endif()

set(ENABLE_OPENMP OFF)
if (NOT PARTHENON_DISABLE_OPENMP)
  find_package(OpenMP COMPONENTS CXX)
  if (NOT OpenMP_FOUND)
    message(FATAL_ERROR "OpenMP is required but couldn't be found. "
    "If you want to build Parthenon without OpenMP, please rerun CMake with -DPARTHENON_DISABLE_OPENMP=ON")
  endif()
  set(ENABLE_OPENMP ON)
  set(Kokkos_ENABLE_OPENMP ON)
endif()

if (Kokkos_ENABLE_CUDA AND TEST_INTEL_OPTIMIZATION)
  message(WARNING
    "Intel optimizer flags may not be passed through NVCC wrapper correctly. "
    "If you encounter problems, please delete your CMake cache "
    "and rerun CMake with -DTEST_INTEL_OPTIMIZATION=OFF.")
endif()

set(ENABLE_HDF5 OFF)
if (NOT PARTHENON_DISABLE_HDF5)
  set(HDF5_PREFER_PARALLEL ${ENABLE_MPI})
  find_package(HDF5 COMPONENTS C)
  if (NOT HDF5_FOUND)
    message(FATAL_ERROR "HDF5 is required but couldn't be found. "
    "If you want to build Parthenon without HDF5, please rerun CMake with -DPARTHENON_DISABLE_HDF5=ON")
  endif()
  set(ENABLE_HDF5 ON)

  if (ENABLE_MPI AND (NOT HDF5_IS_PARALLEL))
    message(FATAL_ERROR "Both MPI and HDF5 are enabled but only a serial version of HDF5 "
    "was found. Please install a parallel version of HDF5 (or point CMake to it by adding its path "
    "to the CMAKE_PREFIX_PATH environment variable), or disable either MPI or HDF5 by rerunning "
    "CMake with -DPARTHENON_DISABLE_MPI=ON or -DPARTHENON_DISABLE_HDF5=ON")
  endif()

  # HDF5 Interface library
  add_library(HDF5_C INTERFACE)
  target_link_libraries(HDF5_C INTERFACE ${HDF5_C_LIBRARIES})
  target_compile_definitions(HDF5_C INTERFACE ${HDF5_C_DEFINITIONS})
  target_include_directories(HDF5_C INTERFACE ${HDF5_C_INCLUDE_DIRS})
endif()

# Kokkos recommendatation resulting in not using default GNU extensions
set(CMAKE_CXX_EXTENSIONS OFF)

# Tell Kokkos to vectorize aggressively
# Kokkos prefers this capitalization for debugging reasons
SET (Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION ON CACHE BOOL
     "Kokkos aggressive vectorization")

# Tell Kokkos we need lambdas in Cuda.
if (Kokkos_ENABLE_CUDA)
  SET (Kokkos_ENABLE_CUDA_LAMBDA ON CACHE BOOL
       "Enable lambda expressions in CUDA")
endif()

# If this is a debug build, set kokkos debug on
if (${CMAKE_BUILD_TYPE} STREQUAL "Debug")
  message(STATUS "Enabling Kokkos debug mode")
  set(Kokkos_ENABLE_DEBUG ON CACHE BOOL "Most general debug settings")
  set(Kokkos_ENABLE_DEBUG_BOUNDS_CHECK ON CACHE BOOL
     "Bounds checking on Kokkos views")
  set(Kokkos_ENABLE_DEBUG_DUALVIEW_MODIFY_CHECK ON CACHE BOOL
      "Sanity checks on Kokkos DualView")
endif()

if (ENABLE_COMPILER_WARNINGS)
  message(STATUS "Enabling -Wall and setting Kokkos_ENABLE_COMPILER_WARNINGS=True")
  set(Kokkos_ENABLE_COMPILER_WARNINGS True CACHE BOOL
      "Make the compiler warn us about things")
  add_compile_options(-Wall)
endif()

# We want Kokkos to be built with C++14, since that's what we're using in
# Parthenon.
set(Kokkos_CXX_STANDARD 14)
if(EXISTS ${Kokkos_ROOT}/CMakeLists.txt)
  add_subdirectory(${Kokkos_ROOT} Kokkos)
elseif(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/external/Kokkos/CMakeLists.txt)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/Kokkos Kokkos)
else()
  find_package(Kokkos 3 REQUIRED)
endif()

# Build Tests and download Catch2
if (${PARTHENON_ENABLE_UNIT_TESTS} OR ${PARTHENON_ENABLE_INTEGRATION_TESTS} OR ${PARTHENON_ENABLE_REGRESSION_TESTS} OR ${PARTHENON_ENABLE_PERFORMANCE_TESTS})

  # if running regression test automatically synch gold standard
  if (${PARTHENON_ENABLE_REGRESSION_TESTS} AND ${REGRESSION_GOLD_STANDARD_SYNC})
    if (EXISTS ${PROJECT_SOURCE_DIR}/tst/regression/gold_standard/current_version)
      file(READ ${PROJECT_SOURCE_DIR}/tst/regression/gold_standard/current_version REGRESSION_GOLD_STANDARD_ON_DISK)
    else()
      set(REGRESSION_GOLD_STANDARD_ON_DISK 0)
    endif()
    if (NOT (${REGRESSION_GOLD_STANDARD_ON_DISK} EQUAL ${REGRESSION_GOLD_STANDARD_VER}))
      message(STATUS "Attempting to download latest gold standard files for regression tests. To disable set REGRESSION_GOLD_STANDARD_SYNC=OFF.")
      file(
        DOWNLOAD https://github.com/lanl/parthenon/releases/download/regression-gold-v${REGRESSION_GOLD_STANDARD_VER}/parthenon_regression_gold_v${REGRESSION_GOLD_STANDARD_VER}.tgz ${PROJECT_SOURCE_DIR}/tst/regression/gold_standard/parthenon_regression_gold_v${REGRESSION_GOLD_STANDARD_VER}.tgz
        EXPECTED_HASH ${REGRESSION_GOLD_STANDARD_HASH}
        )
      execute_process(
        COMMAND ${CMAKE_COMMAND} -E tar xzf ${PROJECT_SOURCE_DIR}/tst/regression/gold_standard/parthenon_regression_gold_v${REGRESSION_GOLD_STANDARD_VER}.tgz
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tst/regression/gold_standard/
        )
      message(STATUS "  Download of gold standard successful.")
    else()
      message(STATUS "Gold standard is up-to-date. No download required.")
    endif()
  endif()

  # Try finding an installed Catch2 first
  find_package(Catch2 2.11.1 QUIET)

  if (NOT Catch2_FOUND)
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

# Globally turn on useful intel and/or nvcc compiler output
if (Kokkos_ENABLE_CUDA)
  if(CHECK_REGISTRY_PRESSURE)
    add_compile_options(-Xptxas=-v)
  endif()
endif()
# Note that these options may not play nice with nvcc wrapper
if (TEST_INTEL_OPTIMIZATION)
  add_compile_options(-fp-model fast=2 -qopt_report5 -vec-threshold0 -qopt_report_phase=vec)
endif()	       

add_subdirectory(src)
add_subdirectory(example)

create_pathenon_coverage_targets()
include(cmake/CheckCopyright.cmake)

# Currently Ctest/Cmake doesn't ensure that tests are not stale
# before running them.
#
# If you add tests with other binaries, add them to the DEPENDS line
add_custom_target(checkit COMMAND ${CMAKE_CTEST_COMMAND}
                  DEPENDS unit_tests)

