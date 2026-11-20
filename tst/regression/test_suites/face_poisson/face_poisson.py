# ========================================================================================
# Parthenon performance portable AMR framework
# Copyright(C) 2026 The Parthenon collaboration
# Licensed under the 3-clause BSD License, see LICENSE file for details
# ========================================================================================
# (C) (or copyright) 2026. Triad National Security, LLC. All rights reserved.
#
# This program was produced under U.S. Government contract 89233218CNA000001 for Los
# Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
# for the U.S. Department of Energy/National Nuclear Security Administration. All rights
# in the program are reserved by Triad National Security, LLC, and the U.S. Department
# of Energy/National Nuclear Security Administration. The Government is granted for
# itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
# license in this material to reproduce, prepare derivative works, distribute copies to
# the public, perform publicly and display publicly, and to permit others to do so.
# ========================================================================================

# [This code was generated with the help of generative AI]
import os
import re
import sys

import utils.test_case

sys.dont_write_bytecode = True

_RES128 = [
    "parthenon/mesh/nx1=128",
    "parthenon/mesh/nx2=128",
    "parthenon/meshblock/nx1=32",
    "parthenon/meshblock/nx2=32",
]
_RES3D = [
    "parthenon/mesh/nx1=32",
    "parthenon/mesh/nx2=32",
    "parthenon/mesh/nx3=32",
    "parthenon/meshblock/nx3=16",
]
# Pure MG does not converge on AMR grids without base block coarsening, since the
# composite levels then have no fine-coarse correction
_AMR = [
    "parthenon/mesh/refinement=static",
    "parthenon/static_refinement0/x1min=-0.6",
    "parthenon/static_refinement0/x1max=0.1",
    "parthenon/static_refinement0/x2min=-0.4",
    "parthenon/static_refinement0/x2max=0.3",
    "parthenon/static_refinement0/level=2",
    "parthenon/mesh/base_block_coarsenings=1",
]


class TestCase(utils.test_case.TestCaseAbs):
    def __init__(self):
        # Each case checks the final error, the iteration count, and the number of owned
        # faces, which is 3 N for N cells on a periodic uniform grid. The iteration counts
        # of the 64^2 and 128^2 cases should be the same.
        #
        # The first input should also be the default --driver_input passed from
        # CMake, since the harness validates that path before Prepare() runs.
        self.cases = [
            {
                "name": "MG, uniform 64^2",
                "args": ["face_poisson/solver=MG"],
                "max_iterations": 16,
                "owned_faces": 3 * 64**2,
            },
            {
                "name": "MG, uniform 128^2",
                "args": ["face_poisson/solver=MG"] + _RES128,
                "max_iterations": 16,
                "owned_faces": 3 * 128**2,
            },
            {
                "name": "MG-CG, uniform 64^2",
                "args": ["face_poisson/solver=CG"],
                "max_iterations": 12,
                "owned_faces": 3 * 64**2,
            },
            {
                "name": "MG-BiCGSTAB, uniform 64^2",
                "args": ["face_poisson/solver=BiCGSTAB"],
                "max_iterations": 13,
                "owned_faces": 3 * 64**2,
            },
            {
                "name": "MG-BiCGSTAB, uniform 32^3",
                "args": ["face_poisson/solver=BiCGSTAB"] + _RES3D,
                "max_iterations": 13,
                "owned_faces": 3 * 32**3,
            },
            {
                "name": "MG-BiCGSTAB, AMR, base block coarsening",
                "args": ["face_poisson/solver=BiCGSTAB"] + _AMR,
                "max_iterations": 22,
                "owned_faces": 76992,
            },
            {
                # rhs = u_exact on non-owned faces covered by a finer neighbor is not
                # consistent with the owned faces, which must not stall the solver. The
                # solution is not known, so the error is not checked.
                "name": "MG-BiCGSTAB, AMR, no base block coarsening, rhs = u_exact",
                "args": ["face_poisson/solver=BiCGSTAB", "face_poisson/rhs=exact"]
                + _AMR[:-1]
                + ["parthenon/mesh/base_block_coarsenings=0"],
                "max_iterations": 22,
                "owned_faces": 76992,
                "max_final_error": float("inf"),
            },
        ]
        for case in self.cases:
            case.setdefault("max_final_error", 1.0e-10)

    def Prepare(self, parameters, step):
        case = self.cases[step - 1]
        parameters.driver_input_path = os.path.join(
            parameters.test_path, "parthinput.face_poisson"
        )
        parameters.driver_cmd_line_args = list(case["args"])
        return parameters

    def Analyse(self, parameters):
        success = True

        cycle_line_re = re.compile(r"^\s*\d+\s+[0-9.eE+\-]+\s*$", re.MULTILINE)
        owned_re = re.compile(r"Owned faces:\s*([0-9]+)")
        final_error_re = re.compile(r"Final rms error:\s*([0-9.eE+\-]+)")

        for case, stdout in zip(self.cases, parameters.stdouts):
            case_success = True
            name = case["name"]
            text = stdout.decode("utf-8", errors="replace")

            iterations = len(cycle_line_re.findall(text))
            if iterations == 0 or iterations > case["max_iterations"]:
                print(
                    f"[{name}] Took {iterations} iterations, expected between 1 and "
                    f"{case['max_iterations']}."
                )
                case_success = False

            m = owned_re.search(text)
            if m is None or int(m.group(1)) != case["owned_faces"]:
                found = m.group(1) if m else "none"
                print(f"[{name}] Owned faces {found}, expected {case['owned_faces']}.")
                case_success = False

            m = final_error_re.search(text)
            if m is None:
                print(f"[{name}] Could not find 'Final rms error' in output.")
                case_success = False
            else:
                final_error = float(m.group(1))
                if not final_error <= case["max_final_error"]:
                    print(
                        f"[{name}] Final rms error too large: "
                        f"{final_error:.6e} > {case['max_final_error']:.6e}"
                    )
                    case_success = False
                elif case_success:
                    print(
                        f"[{name}] PASS: {iterations} iterations, "
                        f"final rms error = {final_error:.6e}"
                    )

            success = success and case_success

        return success
