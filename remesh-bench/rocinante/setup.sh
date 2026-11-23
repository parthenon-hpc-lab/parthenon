#!/bin/bash
# One-time setup on Rocinante, run from a clone of the benchmark branch:
#   git clone -b lroberts36/remesh-bvals-merge-bench \
#     https://github.com/parthenon-hpc-lab/parthenon.git parthenon-bench
#   cd parthenon-bench/remesh-bench && rocinante/setup.sh
# Creates worktrees src-{develop,pr,merge}, patches in the benchmark timers, builds
# build-{develop,pr,merge} and Kokkos Tools (kt-install). Safe to re-run; set JOBS to
# change build parallelism (be polite on a login node, or run inside salloc).
set -euo pipefail
cd "$(dirname "$0")/.."
BENCH=$PWD
REPO=$(git rev-parse --show-toplevel)
JOBS=${JOBS:-16}
source rocinante/env.sh

# Develop at the PR's merge base, the PR head, and this branch (the merge)
declare -A REFS=([develop]=c348b850a [pr]=d1a0ed773 [merge]=HEAD)

echo "== compiler"
CC --version | head -1
tmp=$(mktemp -d)
printf '#include <concepts>\n#include <span>\nint main() { std::span<int> s; return s.size(); }\n' \
  > "$tmp/t.cpp"
if ! CC -std=c++20 "$tmp/t.cpp" -o "$tmp/t" 2> "$tmp/err"; then
  cat "$tmp/err"
  echo "CC cannot compile C++20 (Kokkos 5 needs it); try PrgEnv-gnu in rocinante/env.sh"
  exit 1
fi
rm -rf "$tmp"

echo "== Kokkos submodule"
git -C "$REPO" submodule update --init external/Kokkos

for v in develop pr merge; do
  src=$BENCH/src-$v
  if [ ! -d "$src" ]; then
    echo "== worktree src-$v at ${REFS[$v]}"
    git -C "$REPO" worktree add --detach "$src" "${REFS[$v]}"
    # All three pin the same Kokkos, so share the checkout
    rmdir "$src/external/Kokkos"
    ln -s "$REPO/external/Kokkos" "$src/external/Kokkos"
  fi
  if ! grep -q bench_remesh_timer.hpp "$src/src/mesh/mesh-amr_loadbalance.cpp"; then
    python3 "$BENCH/apply_patch.py" "$src"
  fi
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

if [ ! -f kt-install/lib/libkp_space_time_stack.so ]; then
  echo "== Kokkos Tools"
  [ -d kokkos-tools ] || git clone https://github.com/kokkos/kokkos-tools.git kokkos-tools
  git -C kokkos-tools checkout -q 1c344db04337dcb2aa1566b2a8fee93e9c9f7820
  # Report kernels down to 0.1% of the total instead of 10%, so the remesh kernels show
  sed -i 's/^double output_threshold = 0.1;/double output_threshold = 0.001;/' \
    kokkos-tools/profiling/space-time-stack/kp_space_time_stack.cpp
  cmake -S kokkos-tools -B kokkos-tools/build -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=CC -DCMAKE_C_COMPILER=cc -DCMAKE_INSTALL_PREFIX="$BENCH/kt-install" \
    -DCMAKE_INSTALL_LIBDIR=lib > kt-config.log 2>&1 || { tail -30 kt-config.log; exit 1; }
  cmake --build kokkos-tools/build -j "$JOBS" --target install > kt-build.log 2>&1 \
    || { grep -m 20 -B2 -A5 "error" kt-build.log; exit 1; }
fi

echo "== done"
ls -l build-*/example/advection/advection-example kt-install/lib/libkp_space_time_stack.so
