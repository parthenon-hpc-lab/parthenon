# Source from bash: module environment for building and running the remesh benchmark.
# PrgEnv-intel is the site default (icpx behind the CC wrapper). Kokkos 5 needs C++20;
# setup.sh checks that the compiler and its libstdc++ provide it. If not, switch to
# PrgEnv-gnu here (and rebuild from scratch).
module load PrgEnv-intel
module load cray-mpich
module load craype-x86-spr
module load cmake/3.31.6
module load cray-python
module load git 2>/dev/null || true
# Statically linked executables would defeat the Kokkos Tools dlopen
export CRAYPE_LINK_TYPE=dynamic
