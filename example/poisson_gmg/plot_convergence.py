# =========================================================================================
# (C) (or copyright) 2020-2024. Triad National Security, LLC. All rights reserved.
#
# This program was produced under U.S. Government contract 89233218CNA000001 for Los
# Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
# for the U.S. Department of Energy/National Nuclear Security Administration. All rights
# in the program are reserved by Triad National Security, LLC, and the U.S. Department
# of Energy/National Nuclear Security Administration. The Government is granted for
# itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
# license in this material to reproduce, prepare derivative works, distribute copies to
# the public, perform publicly and display publicly, and to permit others to do so.
# =========================================================================================

import numpy as np
import glob
import matplotlib.pyplot as plt 
import subprocess

plt.style.use('tableau-colorblind10')

solver = "BiCGSTAB"
difco = 1e6
for bound_pro in ["Constant", "Linear"]:
  for interior_pro in ["Constant", "OldLinear"]:

    p = subprocess.run(["./poisson-gmg-example", "-i", "parthinput.poisson",
                     "poisson/solver=" + solver, 
                     "poisson/interior_D=" + str(difco), 
                     "poisson/prolongation=" + bound_pro, 
                     "poisson/solver_params/prolongation=" + interior_pro], capture_output = True, text = True)
    dat = np.genfromtxt(p.stdout.splitlines())

    plt.semilogy(dat[:, 0], dat[:, 1], label=solver + "_" + str(difco) + "_" + bound_pro + "_" + interior_pro)


plt.legend(loc = 'upper right')
plt.ylim([1.e-14, 1e2])
plt.xlim([0, 40])
plt.xlabel("# of V-cycles")
plt.ylabel("RMS Residual")
plt.savefig("convergence_1e6.pdf")