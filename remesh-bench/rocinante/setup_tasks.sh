#!/bin/bash
# Setup for the remesh-tasks comparison (bench.py --machine rocinante-tasks), run from
# remesh-bench/ after rocinante/setup.sh (which builds Kokkos Tools):
#   rocinante/setup_tasks.sh
# Creates unpatched worktrees src-{dev,lanes,tasks} and builds build-{dev,lanes,tasks};
# timing comes from space-time-stack regions, not the benchmark timer patch. Re-running
# moves src-tasks to the latest lroberts36/remesh-tasks and rebuilds.
set -euo pipefail
cd "$(dirname "$0")/.."
BENCH=$PWD
REPO=$(git rev-parse --show-toplevel)
JOBS=${JOBS:-16}
source rocinante/env.sh

git -C "$REPO" fetch origin lroberts36/remesh-tasks lroberts36/faster-coalesced-comms develop
# develop at the merge base of remesh-tasks, lanes (faster-coalesced-comms, the base of
# remesh-tasks), and remesh-tasks
declare -A REFS=([dev]=e28dd5720 [lanes]=4653525c4 [tasks]=origin/lroberts36/remesh-tasks)

git -C "$REPO" submodule update --init external/Kokkos
for v in dev lanes tasks; do
  src=$BENCH/src-$v
  if [ ! -d "$src" ]; then
    echo "== worktree src-$v at ${REFS[$v]}"
    git -C "$REPO" worktree add --detach "$src" "${REFS[$v]}"
    # All pin the same Kokkos, so share the checkout
    rmdir "$src/external/Kokkos"
    ln -s "$REPO/external/Kokkos" "$src/external/Kokkos"
  else
    git -C "$src" checkout -q --detach "${REFS[$v]}"
  fi
  echo "== src-$v: $(git -C "$src" log --oneline -1)"
  if [ ! -f "build-$v/CMakeCache.txt" ]; then
    echo "== configure build-$v"
    cmake -S "$src" -B "build-$v" -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_COMPILER=CC -DCMAKE_C_COMPILER=cc \
      -DKokkos_ENABLE_SERIAL=ON -DKokkos_ARCH_SPR=ON \
      -DPARTHENON_DISABLE_HDF5=ON -DPARTHENON_DISABLE_OPENPMD=ON \
      -DPARTHENON_ENABLE_TESTING=OFF -DBUILD_TESTING=OFF > "build-$v.cfg.log" 2>&1 \
      || { tail -30 "build-$v.cfg.log"; exit 1; }
  fi
  echo "== build build-$v"
  cmake --build "build-$v" --target advection-example -j "$JOBS" > "build-$v.log" 2>&1 \
    || { grep -m 20 -B2 -A5 "error" "build-$v.log"; exit 1; }
done

[ -f kt-install/lib/libkp_space_time_stack.so ] \
  || echo "!! Kokkos Tools missing: run rocinante/setup.sh first"
echo "== done"
ls -l build-{dev,lanes,tasks}/example/advection/advection-example
