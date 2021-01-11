#=========================================================================================
# (C) (or copyright) 2021. Triad National Security, LLC. All rights reserved.
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

# Function will check that the specified modules are available to the python interpreter
# If they are not cmake will throw an error indicating which module not available
function(required_python_modules_found module_list)

  unset(MISSING_MODULES)
  if(${Python3_Interpreter_FOUND})
    set(IMPORT_ERROR 0)
    foreach(module IN LISTS module_list )
      execute_process(COMMAND ${Python3_EXECUTABLE} -c "import ${module}"
        RESULT_VARIABLE IMPORT_MODULE ERROR_QUIET)
    
      if(NOT ${IMPORT_MODULE} EQUAL 0)
        set(IMPORT_ERROR 1)
        list(APPEND MISSING_MODULES ${module})
      endif()
    endforeach()

    if (IMPORT_ERROR)
      message(FATAL_ERROR "Required python module(s) ${MISSING_MODULES} not found.") 
    endif()
  endif()
endfunction()
