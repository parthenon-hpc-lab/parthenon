#=========================================================================================
# (C) (or copyright) 2020-2021. Triad National Security, LLC. All rights reserved.
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
file(DOWNLOAD https://codecov.io/bash ${COVERAGE_PATH}/CombinedCoverage/script.coverage)

file(READ ${COVERAGE_PATH}/CombinedCoverage/script.coverage COVERAGE_FILE_CONTENT)
string(REGEX MATCH "VERSION=\"[0-9.].[0-9.].[0-9.]\"" VERSION_NUM ${COVERAGE_FILE_CONTENT})
STRING(REGEX MATCH "\"(.+)\"" VERSION_NUM ${VERSION_NUM})
STRING(REPLACE "\"" "" VERSION_NUM ${VERSION_NUM})
message("VERSION NUM ${VERSION_NUM}")
file(DOWNLOAD https://raw.githubusercontent.com/codecov/codecov-bash/${VERSION_NUM}/SHA256SUM ${COVERAGE_PATH}/CombinedCoverage/trusted_hashes)
file(READ ${COVERAGE_PATH}/CombinedCoverage/trusted_hashes HASH_FILE_CONTENT)

string(REGEX REPLACE "\n$" "" HASH_FILE_CONTENT "${HASH_FILE_CONTENT}")
string(REGEX REPLACE "  " ";" HASH_FILE_CONTENT "${HASH_FILE_CONTENT}")

list(GET HASH_FILE_CONTENT 0 TRUSTED_HASH)

file(SHA256 ${COVERAGE_PATH}/CombinedCoverage/script.coverage HASH_VAL_FILE)

if( NOT ${HASH_VAL_FILE} STREQUAL ${TRUSTED_HASH} )
  message(FATAL_ERROR "Untrusted bash codecoverage script detected.")
endif()

