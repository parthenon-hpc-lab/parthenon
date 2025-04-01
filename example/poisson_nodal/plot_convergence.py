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

plt.style.use("tableau-colorblind10")

solver = "BiCGSTAB"
solver_lbl = "BCGS"
difco = 1e6
refine = True

for bound_pro in ["Constant", "Linear"]:
    for interior_pro in ["Constant", "Linear", "SplitLin", "Kwak"]:
        command = ["./poisson-gmg-example", "-i", "parthinput.poisson"]
        command.append("poisson/solver=" + solver)
        command.append("poisson/interior_D=" + str(difco))
        command.append("poisson/boundary_prolongation=" + bound_pro)
        if interior_pro == "SplitLin":
            command.append("poisson/solver_params/prolongation=Default")
        else:
            command.append("poisson/interior_prolongation=" + interior_pro)
            command.append("poisson/solver_params/prolongation=User")

        if refine:
            command.append("parthenon/static_refinement0/x1min=-1.0")
            command.append("parthenon/static_refinement0/x1max=-0.75")
            command.append("parthenon/static_refinement0/x2min=-1.0")
            command.append("parthenon/static_refinement0/x2max=-0.75")
            command.append("parthenon/static_refinement0/level=3")

        p = subprocess.run(command, capture_output=True, text=True)
        lines = p.stdout.splitlines()
        # Ignore any initialization junk that gets spit out earlier from adding parameters
        idx = lines.index("# [0] v-cycle")
        dat = np.genfromtxt(lines[idx:])
        label = "{}_{}".format(solver_lbl, interior_pro)
        if refine:
            label = "{}_{}_Bnd{}".format(solver_lbl, interior_pro, bound_pro)
        plt.semilogy(
            dat[:, 0],
            dat[:, 1],
            label=label.replace("Constant", "Const").replace("Linear", "Lin"),
        )


plt.legend()
plt.ylim([1.0e-14, 1e4])
plt.xlim([0, 40])
plt.xlabel("# of V-cycles")
plt.ylabel("RMS Residual")
plt.title("$D_{int} = 10^6$ w/ Refinement")
plt.savefig("convergence_1e6.pdf")
