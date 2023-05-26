#=========================================================================================
# Parthenon performance portable AMR framework
# Copyright(C) 2020-2023 The Parthenon collaboration
# Licensed under the 3-clause BSD License, see LICENSE file for details
#=========================================================================================
# (C) (or copyright) 2020-2023. Triad National Security, LLC. All rights reserved.
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

cmake_minimum_required(VERSION 3.16)

# Imports machine-specific configuration
include(cmake/MachineCfg.cmake)

project(parthenon VERSION 0.8.0 LANGUAGES C CXX)

if (${CMAKE_VERSION} VERSION_GREATER_EQUAL 3.19.0)
  cmake_policy(SET CMP0110 NEW)
endif()

include(CTest)

# Compile time constants

# Compile Options
option(PARTHENON_SINGLE_PRECISION "Run in single precision" OFF)
option(PARTHENON_DISABLE_MPI "MPI is enabled by default if found, set this to True to disable MPI" OFF)
option(PARTHENON_ENABLE_HOST_COMM_BUFFERS "CUDA/HIP Only: Allocate communication buffers on host (may be slower)" OFF)
option(PARTHENON_DISABLE_OPENMP "OpenMP is enabled by default if found, set this to True to disable OpenMP" OFF)
option(PARTHENON_DISABLE_HDF5 "HDF5 is enabled by default if found, set this to True to disable HDF5" OFF)
option(PARTHENON_DISABLE_HDF5_COMPRESSION "HDF5 compression is enabled by default, set this to True to disable compression in HDF5 output/restart files" OFF)
option(PARTHENON_DISABLE_SPARSE "Sparse capability is enabled by default, set this to True to compile-time disable all sparse capability" OFF)
option(PARTHENON_ENABLE_ASCENT "Enable Ascent for in situ visualization and analysis" OFF)
option(PARTHENON_LINT_DEFAULT "Linting is turned off by default, use the \"lint\" target or set \
this to True to enable linting in the default target" OFF)
option(PARTHENON_COPYRIGHT_CHECK_DEFAULT "Copyright check is turned off by default, use the \
\"check-copyright\" target or set this to True to check copyright in the default target" OFF)
option(PARTHENON_DISABLE_EXAMPLES "Example drivers are built by default, set this to True to disable, \
note if regression tests are built some, drivers needed by the regression tests will still be built." OFF)
option(PARTHENON_ENABLE_TESTING "Default for all test types." ${BUILD_TESTING})
option(PARTHENON_ENABLE_PYTHON_MODULE_CHECK "Check if local python version contains all modules required for running tests." ${PARTHENON_ENABLE_TESTING})
option(PARTHENON_ENABLE_GPU_MPI_CHECKS "Checks if possible that the mpi num of procs and the number\
of gpu devices detected are appropriate." ${PARTHENON_ENABLE_TESTING})
option(PARTHENON_ENABLE_UNIT_TESTS "Enable unit tests" ${PARTHENON_ENABLE_TESTING})
option(PARTHENON_ENABLE_INTEGRATION_TESTS "Enable integration tests" ${PARTHENON_ENABLE_TESTING})
option(PARTHENON_ENABLE_PERFORMANCE_TESTS "Enable performance tests" ${PARTHENON_ENABLE_TESTING})
option(PARTHENON_ENABLE_REGRESSION_TESTS "Enable regression tests" ${PARTHENON_ENABLE_TESTING})
option(ENABLE_COMPILER_WARNINGS "Enable compiler warnings" OFF)
option(CHECK_REGISTRY_PRESSURE "Check the registry pressure for Kokkos CUDA kernels" OFF)
option(TEST_INTEL_OPTIMIZATION "Test intel optimization and vectorization" OFF)
option(TEST_ERROR_CHECKING "Enables the error checking unit test. This test will FAIL" OFF)
option(CODE_COVERAGE "Enable code coverage reporting" OFF)

# Default to an external Kokkos package if the submodule is not populated
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/external/Kokkos/CMakeLists.txt" AND NOT EXISTS "${Kokkos_ROOT}/CMakeLists.txt")
    option(PARTHENON_IMPORT_KOKKOS "If ON, attempt to link to an external Kokkos library. If OFF, build Kokkos from source and package with Parthenon" ON)
else()
    option(PARTHENON_IMPORT_KOKKOS "If ON, attempt to link to an external Kokkos library. If OFF, build Kokkos from source and package with Parthenon" OFF)
endif()

include(cmake/Format.cmake)
include(cmake/Lint.cmake)

# regression test reference data
set(REGRESSION_GOLD_STANDARD_VER 16 CACHE STRING "Version of gold standard to download and use")
set(REGRESSION_GOLD_STANDARD_HASH
  "SHA512=33f5df050dbca10a4e7ee1788007f3bd00fe549b8f9e0ae8e4175b76f1cdb28190d64c9eecd9daecdcd880aade8614e8b48d534a96a23b59d7356638ef6045f3"
  CACHE STRING "Hash of default gold standard file to download")
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

# Require C++17's std::filesystem
# This is fully standard, but links a separate library when using old compilers
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(Filesystem REQUIRED COMPONENTS Experimental Final)

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
  # Using Host OpenMP and Cuda with nvcc currently does not compile with C++17 standard.
  # Also there is no proper support for two separate execution spaces in Parthenon right now.
  # We may need to revisit this logic when OpenMP target for devices will be used in the future.
  if (Kokkos_ENABLE_CUDA OR Kokkos_ENABLE_HIP)
    message(STATUS "Parthenon is not using Host OpenMP because Cuda or HIP is enabled")
  else()
    find_package(OpenMP COMPONENTS CXX)
    if (NOT OpenMP_FOUND)
      message(FATAL_ERROR "OpenMP is required but couldn't be found. "
      "If you want to build Parthenon without OpenMP, please rerun CMake with -DPARTHENON_DISABLE_OPENMP=ON")
    endif()
    set(ENABLE_OPENMP ON)
    set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "Allow Kokkos to use OpenMP as execution space.")
  endif()
endif()

set(ENABLE_SPARSE ON)
if (PARTHENON_DISABLE_SPARSE)
  set(ENABLE_SPARSE OFF)
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
    "to the CMAKE_PREFIX_PATH environment variable or by setting the HDF5_ROOT variable), "
    "or disable either MPI or HDF5 by rerunning "
    "CMake with -DPARTHENON_DISABLE_MPI=ON or -DPARTHENON_DISABLE_HDF5=ON")
  endif()

  if ((NOT ENABLE_MPI) AND HDF5_IS_PARALLEL)
    message(FATAL_ERROR "MPI is disabled and HDF5 is enabled but only a parallel version of HDF5 "
    "was found. Please install a serial version of HDF5 (or point CMake to it by adding its path "
    "to the CMAKE_PREFIX_PATH environment variable or by setting the HDF5_ROOT variable), or "
    "either enable MPI or disable HDF5 by rerunning "
    "CMake with -DPARTHENON_DISABLE_MPI=OFF or -DPARTHENON_DISABLE_HDF5=ON")
  endif()

  if ((NOT PARTHENON_DISABLE_HDF5_COMPRESSION) AND ENABLE_MPI)
    # make sure we use an HDF5 version that doesn't have a known bug when used with compression and
    # parallel I/O. See:
    # https://forum.hdfgroup.org/t/crash-when-writing-parallel-compressed-chunks/6186
    #
    # It appears that the bug is fixed in 1.10.7 and 1.12.0, assume that later versions are ok as
    # well, so we check that we have at least 1.10.7
    if (HDF5_VERSION VERSION_LESS "1.10.7")
      message(FATAL_ERROR "This version of HDF5 has a known bug that will crash when trying to "
      "write compressed datasets with MPI parallel I/O. Please use HDF5 version 1.10.7 or later "
      "or turn off either HDF5 compression or MPI by rerunning CMake with "
      "-DPARTHENON_DISABLE_HDF5_COMPRESSION=ON or -DPARTHENON_DISABLE_MPI=ON.\nNOTE: Using HDF5 "
      "without compression can result in much larger HDF5 output and restart files as well as "
      "degraded I/O performance.")
    endif()
  endif()

  # HDF5 Interface library
  add_library(HDF5_C INTERFACE)
  target_link_libraries(HDF5_C INTERFACE ${HDF5_C_LIBRARIES})
  target_compile_definitions(HDF5_C INTERFACE ${HDF5_C_DEFINITIONS})
  target_include_directories(HDF5_C INTERFACE ${HDF5_C_INCLUDE_DIRS})

  install(TARGETS HDF5_C EXPORT parthenonTargets)
endif()

# Kokkos recommendatation resulting in not using default GNU extensions
set(CMAKE_CXX_EXTENSIONS OFF)

# Tell Kokkos to vectorize aggressively
# Kokkos prefers this capitalization for debugging reasons
SET (Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION ON CACHE BOOL
     "Kokkos aggressive vectorization")

# Check that gpu devices are actually detected
set(NUM_GPU_DEVICES_PER_NODE "1" CACHE STRING "Number of gpu devices to use when testing if built with Kokkos_ENABLE_CUDA")
set(NUM_OMP_THREADS_PER_RANK "1" CACHE STRING "Number of threads to use when testing if built with Kokkos_ENABLE_OPENMP")
if (Kokkos_ENABLE_CUDA)
  # Tell Kokkos we need lambdas in Cuda.
  SET (Kokkos_ENABLE_CUDA_LAMBDA ON CACHE BOOL
       "Enable lambda expressions in CUDA")
  if ( "${PARTHENON_ENABLE_GPU_MPI_CHECKS}" )
     configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/CTestCustom.cmake.in ${CMAKE_BINARY_DIR}/CTestCustom.cmake @ONLY)
  endif()
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

set(Kokkos_ENABLE_DEPRECATED_CODE_3 OFF CACHE BOOL "No need for old/unused code.")

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)

if (NOT TARGET Kokkos::kokkos)
  if (PARTHENON_IMPORT_KOKKOS)
    find_package(Kokkos 3)
    if (NOT Kokkos_FOUND)
      unset(PARTHENON_IMPORT_KOKKOS CACHE)
      message(FATAL_ERROR "Could not find external Kokkos. Consider importing a Kokkos installation into your environment or disabling external Kokkos with e.g. -DPARTHENON_IMPORT_KOKKOS=OFF")
    endif()
  else()
    if (EXISTS ${Kokkos_ROOT}/CMakeLists.txt)
      add_subdirectory(${Kokkos_ROOT} Kokkos)
      message(STATUS "Using Kokkos source from Kokkos_ROOT=${Kokkos_ROOT}")
    elseif(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/external/Kokkos/CMakeLists.txt)
      add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/Kokkos Kokkos)
      message(STATUS "Using Kokkos source from Parthenon submodule at ${CMAKE_CURRENT_SOURCE_DIR}/external/Kokkos")
    else()
      message(FATAL_ERROR "Could not find Kokkos source. Consider running `git submodule update --init`, providing the path to a Kokkos source directory with Kokkos_ROOT, or setting PARTHENON_IMPORT_KOKKOS=ON to link to an external Kokkos installation.")
    endif()
  endif()
endif()

if (Kokkos_ENABLE_SYCL)
  message(WARNING
    "SYCL backend is currently NOT tested in Parthenon due to lack of access to hardware. "
    "It may or may not work. Use at your own risk. "
    "Please get in contact for support."
  )
endif()

if (PARTHENON_ENABLE_HOST_COMM_BUFFERS)
  if (NOT Kokkos_ENABLE_CUDA AND NOT Kokkos_ENABLE_HIP)
    message(FATAL_ERROR "Host pinned buffers for MPI communication are supported only for CUDA and HIP backends.")
  endif()
endif()

# Build Tests and download Catch2
if (PARTHENON_ENABLE_UNIT_TESTS OR PARTHENON_ENABLE_INTEGRATION_TESTS OR PARTHENON_ENABLE_REGRESSION_TESTS OR PARTHENON_ENABLE_PERFORMANCE_TESTS)

  # if running regression test automatically synch gold standard
  if (PARTHENON_ENABLE_REGRESSION_TESTS AND REGRESSION_GOLD_STANDARD_SYNC)
    if (EXISTS ${PROJECT_SOURCE_DIR}/tst/regression/gold_standard/current_version)
      file(READ ${PROJECT_SOURCE_DIR}/tst/regression/gold_standard/current_version REGRESSION_GOLD_STANDARD_ON_DISK)
    else()
      set(REGRESSION_GOLD_STANDARD_ON_DISK 0)
    endif()
    if (NOT (${REGRESSION_GOLD_STANDARD_ON_DISK} EQUAL ${REGRESSION_GOLD_STANDARD_VER}))
      message(STATUS "Attempting to download latest gold standard files for regression tests. To disable set REGRESSION_GOLD_STANDARD_SYNC=OFF.")
      file(
        DOWNLOAD https://github.com/parthenon-hpc-lab/parthenon/releases/download/regression-gold-v${REGRESSION_GOLD_STANDARD_VER}/parthenon_regression_gold_v${REGRESSION_GOLD_STANDARD_VER}.tgz ${PROJECT_SOURCE_DIR}/tst/regression/gold_standard/parthenon_regression_gold_v${REGRESSION_GOLD_STANDARD_VER}.tgz
        EXPECTED_HASH ${REGRESSION_GOLD_STANDARD_HASH}
        )
      execute_process(
        COMMAND ${CMAKE_COMMAND} -E tar xzf ${PROJECT_SOURCE_DIR}/tst/regression/gold_standard/parthenon_regression_gold_v${REGRESSION_GOLD_STANDARD_VER}.tgz
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tst/regression/gold_standard/
        )
      message(STATUS "  Download of gold standard successful. Now using version ${REGRESSION_GOLD_STANDARD_VER}")
    else()
      message(STATUS "Gold standard is up-to-date (version ${REGRESSION_GOLD_STANDARD_VER}). No download required.")
    endif()


    # Throw a warning if the number of processors available is less than the number specified for running the tests.
    include(ProcessorCount)
    ProcessorCount(N)
    if( "${NUM_MPI_PROC_TESTING}" GREATER "${N}")
      message(WARNING "Consider changing the number of MPI processors used in testing, currently "
        "NUM_MPI_PROC_TESTING is set to ${NUM_MPI_PROC_TESTING} but cmake has only detected a"
        " total of ${N} available processors")
    endif()
  endif()

  if (NOT TARGET catch2_define)
    # Try finding an installed Catch2 first
    find_package(Catch2 2.13.8 QUIET)

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
  endif()

  include(Catch)
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

if (PARTHENON_ENABLE_ASCENT)
  find_package(Ascent REQUIRED NO_DEFAULT_PATH)
endif()

add_subdirectory(src)
add_subdirectory(example)
add_subdirectory(benchmarks)

create_pathenon_coverage_targets()
include(cmake/CheckCopyright.cmake)

configure_file("${PROJECT_SOURCE_DIR}/cmake/parthenonConfig.cmake.in" "${PROJECT_BINARY_DIR}/cmake/parthenonConfig.cmake" @ONLY)

# Currently Ctest/Cmake doesn't ensure that tests are not stale
# before running them.
#
# If you add tests with other binaries, add them to the DEPENDS line
add_custom_target(checkit COMMAND ${CMAKE_CTEST_COMMAND}
                  DEPENDS unit_tests)
