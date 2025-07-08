#========================================================================================
# Parthenon performance portable AMR framework
# Copyright(C) 2020-2025 The Parthenon collaboration
# Licensed under the 3-clause BSD License, see LICENSE file for details
#========================================================================================
# (C) (or copyright) 2020-2025. Triad National Security, LLC. All rights reserved.
#
# This program was produced under U.S. Government contract 89233218CNA000001 for Los
# Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
# for the U.S. Department of Energy/National Nuclear Security Administration. All rights
# in the program are reserved by Triad National Security, LLC, and the U.S. Department
# of Energy/National Nuclear Security Administration. The Government is granted for
# itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
# license in this material to reproduce, prepare derivative works, distribute copies to
# the public, perform publicly and display publicly, and to permit others to do so.
#========================================================================================
# Originally based on code by Jonathan Hamberg
# https://gitlab.com/jhamberg/cmake-examples/-/tree/master/cmake
set(CURRENT_LIST_DIR ${CMAKE_CURRENT_LIST_DIR})
if (NOT DEFINED pre_configure_dir)
    set(pre_configure_dir ${CMAKE_SOURCE_DIR}/src)
endif ()

if (NOT DEFINED post_configure_dir)
    set(post_configure_dir ${CMAKE_BINARY_DIR}/generated)
endif ()

set(pre_configure_file ${pre_configure_dir}/provenance.cpp.in)
set(post_configure_file ${post_configure_dir}/provenance.cpp)

function(CheckGitWrite git_hash)
    file(WRITE ${CMAKE_BINARY_DIR}/git-state-parthenon.txt ${git_hash})
endfunction()

function(CheckGitRead git_hash)
    if (EXISTS ${CMAKE_BINARY_DIR}/git-state-parthenon.txt)
        file(STRINGS ${CMAKE_BINARY_DIR}/git-state-parthenon.txt CONTENT)
        LIST(GET CONTENT 0 var)

        set(${git_hash} ${var} PARENT_SCOPE)
    endif ()
endfunction()

function(CheckGitVersion)
    # Get the latest abbreviated commit hash of the working branch
    execute_process(
        COMMAND git log -1 --format=%h
        WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
        OUTPUT_VARIABLE PARTH_GIT_HASH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        )

    # Get the git branch
    execute_process(
        COMMAND git rev-parse --abbrev-ref HEAD
        WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
        OUTPUT_VARIABLE PARTH_GIT_BRANCH
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    CheckGitRead(PARTH_GIT_HASH_CACHE)
    if (NOT EXISTS ${post_configure_dir})
        message("WHAT1")
        file(MAKE_DIRECTORY ${post_configure_dir})
    endif ()

    if (NOT EXISTS ${post_configure_dir}/provenance.hpp)
        message("WHAT")
        file(COPY ${pre_configure_dir}/provenance.hpp DESTINATION ${post_configure_dir})
    endif()

    if (NOT DEFINED PARTH_GIT_HASH_CACHE)
        set(PARTH_GIT_HASH_CACHE "INVALID")
    endif ()

    # Only update the provenance.cpp if the hash has changed. This will
    # prevent us from rebuilding the project more than we need to.
    if (NOT ${PARTH_GIT_HASH} STREQUAL ${PARTH_GIT_HASH_CACHE} OR NOT EXISTS ${post_configure_file})
        # Set the PARTH_GIT_HASH_CACHE variable the next build won't have
        # to regenerate the source file.
        message("should i be here")
        message("HASH ${PARTH_GIT_HASH}")
        message("HASH CACHE ${PARTH_GIT_HASH_CACHE}")
        CheckGitWrite(${PARTH_GIT_HASH})

        configure_file(${pre_configure_file} ${post_configure_file} @ONLY)
    endif ()

endfunction()

function(CheckGitSetup)

    add_custom_target(ParthenonAlwaysCheckGit COMMAND ${CMAKE_COMMAND}
        -DRUN_CHECK_GIT_VERSION=1
        -Dpre_configure_dir=${pre_configure_dir}
        -Dpost_configure_file=${post_configure_dir}
        -DPARTH_GIT_HASH_CACHE=${PARTH_GIT_HASH_CACHE}
        -P ${CURRENT_LIST_DIR}/Provenance.cmake
        BYPRODUCTS ${post_configure_file}
        )

    add_library(git_version ${CMAKE_BINARY_DIR}/generated/provenance.cpp)
    target_include_directories(git_version PUBLIC ${CMAKE_BINARY_DIR}/generated)
    add_dependencies(git_version ParthenonAlwaysCheckGit)

    CheckGitVersion()
endfunction()

# This is used to run this function from an external cmake process.
if (RUN_CHECK_GIT_VERSION)
    CheckGitVersion()
endif ()

# Other information:
# Compiler
# build options
# timestamp
string(TIMESTAMP PARTH_BUILD_TIMESTAMP "%Y-%m-%d %H:%M:%S %Z")
set(PARTH_COMPILER "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
set(PARTH_COMPILER_VERSION ${CMAKE_CXX_COMPILER_VERSION})
set(PARTH_OPTIMIZATION ${CMAKE_BUILD_TYPE})
set(PARTH_ARCH ${CMAKE_HOST_SYSTEM_PROCESSOR})
#configure_file(${pre_configure_file} ${post_configure_file} @ONLY)
