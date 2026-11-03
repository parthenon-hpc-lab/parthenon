#=========================================================================================
# (C) (or copyright) 2020-2022. Triad National Security, LLC. All rights reserved.
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
# This file was made in part with generative AI

# Formatting is done by scripts/format.sh, which uses the tool versions pinned in
# .pre-commit-config.yaml if pre-commit is available and locally installed tools
# otherwise. In the latter case, CLANG_FORMAT and BLACK (e.g., from a machine config)
# are used if set.
set(_format_env)
foreach(_tool CLANG_FORMAT BLACK)
  if (${_tool})
    list(APPEND _format_env "${_tool}=${${_tool}}")
  endif()
endforeach()

add_custom_target(format
  COMMAND ${CMAKE_COMMAND} -E env ${_format_env} ${PROJECT_SOURCE_DIR}/scripts/format.sh
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
  VERBATIM)
