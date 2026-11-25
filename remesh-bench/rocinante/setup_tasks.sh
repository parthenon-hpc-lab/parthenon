#!/bin/bash
# Setup for the remesh-tasks comparison (bench.py --machine rocinante-tasks), run from
# remesh-bench/ after rocinante/setup.sh (which builds Kokkos Tools):
#   rocinante/setup_tasks.sh --fetch-only     # login node: GitHub fetch and worktrees
#   JOBS=64 srun -A xd -p standard -N 1 -n 1 --exclusive -t 1:00:00 \
#     rocinante/setup_tasks.sh --no-fetch     # compute node: builds
# (with no argument it does both, for machines where compute nodes reach GitHub).
# Creates unpatched worktrees src-{dev,lanes,tasks} and builds build-{dev,lanes,tasks};
# timing comes from space-time-stack regions, not the benchmark timer patch. Re-running
# moves src-tasks to the latest lroberts36/remesh-tasks and rebuilds.
set -euo pipefail
cd "$(dirname "$0")/.."
BENCH=$PWD
REPO=$(git rev-parse --show-toplevel)
JOBS=${JOBS:-16}
MODE=${1:-all}
case "$MODE" in all | --fetch-only | --no-fetch) ;; *) echo "unknown option $MODE"; exit 1 ;; esac
source rocinante/env.sh

if [ "$MODE" != --no-fetch ]; then
  # Explicit refspecs, so the remote-tracking refs exist even in a single-branch clone
  for b in lroberts36/remesh-tasks lroberts36/faster-coalesced-comms develop; do
    git -C "$REPO" fetch origin "+refs/heads/$b:refs/remotes/origin/$b"
  done
  git -C "$REPO" submodule update --init external/Kokkos
fi
# develop at the merge base of remesh-tasks, lanes (faster-coalesced-comms, the base of
# remesh-tasks), and remesh-tasks
ref_of() {
  case $1 in
    dev) echo e28dd5720 ;;
    lanes) echo 4653525c4 ;;
    tasks) echo origin/lroberts36/remesh-tasks ;;
  esac
}

for v in dev lanes tasks; do
  src=$BENCH/src-$v
  if [ ! -d "$src" ]; then
    echo "== worktree src-$v at $(ref_of $v)"
    git -C "$REPO" worktree add --detach "$src" "$(ref_of $v)"
    # All pin the same Kokkos, so share the checkout
    rmdir "$src/external/Kokkos"
    ln -s "$REPO/external/Kokkos" "$src/external/Kokkos"
  else
    git -C "$src" checkout -q --detach "$(ref_of $v)"
  fi
  echo "== src-$v: $(git -C "$src" log --oneline -1)"
  [ "$MODE" = --fetch-only ] && continue
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

[ "$MODE" = --fetch-only ] && { echo "== fetched; now build on a compute node with --no-fetch"; exit 0; }
[ -f kt-install/lib/libkp_space_time_stack.so ] \
  || echo "!! Kokkos Tools missing: run rocinante/setup.sh first"
echo "== done"
ls -l build-{dev,lanes,tasks}/example/advection/advection-example
