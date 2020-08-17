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

find_program( NVIDIA_SMI NAMES nvidia-smi)

set(NUM_GPUS_DETECTED 0)
if( NVIDIA_SMI )
  execute_process( COMMAND ${NVIDIA_SMI} -L OUTPUT_VARIABLE SMI_OUTPUT)
  set(GPU_ID "0")
  set(GPU_DETECT "GPU ${GPU_ID}: ")
  while("${SMI_OUTPUT}" MATCHES "${GPU_DETECT}")
    math(EXPR NUM_GPUS_DETECTED "${NUM_GPUS_DETECTED} + 1")
    math(EXPR GPU_ID "${GPU_ID} + 1")
    set(GPU_DETECT "GPU ${GPU_ID}: ")
  endwhile()
endif()
