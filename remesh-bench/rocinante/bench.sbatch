#!/bin/bash
# Remesh benchmark sweep on Rocinante. Submit from remesh-bench/:
#   sbatch -N 4 -p <partition> [-A <account>] rocinante/bench.sbatch [bench.py args]
#   sbatch -N 1 -t 0:30:00 rocinante/bench.sbatch --only b32 --ranks 8 --reps 1  # smoke
# Cases needing more nodes than the allocation are skipped; extra arguments go to
# bench.py (e.g. --variants develop pr_rw merge merge_pt merge_ws merge_ap).
#SBATCH -J remesh-bench
#SBATCH -N 4
#SBATCH -t 08:00:00
#SBATCH --exclusive
#SBATCH -o remesh-bench-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source rocinante/env.sh

echo "== $(date) job ${SLURM_JOB_ID}: ${SLURM_NNODES} nodes ${SLURM_JOB_NODELIST}"
module list 2>&1
# One core (two hyperthreads) per rank; check what the progress thread will see
srun -N 1 -n 4 -c 2 --cpu-bind=cores \
  bash -c 'echo "rank $SLURM_PROCID on $(hostname): $(grep Cpus_allowed_list /proc/self/status)"'

python3 bench.py --machine rocinante --reps 2 --keep-going \
  --outdir "results-rocinante-${SLURM_JOB_ID}" "$@"
# Peak RSS of the busiest rank in each run (the RSS watchdog cannot see srun ranks)
sacct -j "${SLURM_JOB_ID}" -P --format=JobID,JobName%40,NTasks,MaxRSS,MaxRSSNode,Elapsed \
  > "results-rocinante-${SLURM_JOB_ID}/sacct.txt" || true
echo "== $(date) done"
