# ========================================================================================
# Parthenon performance portable AMR framework
# Copyright(C) 2020 The Parthenon collaboration
# Licensed under the 3-clause BSD License, see LICENSE file for details
# ========================================================================================
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
# ========================================================================================

# [This code was generated with the help of generative AI]
import os
import re
import sys

import utils.test_case

sys.dont_write_bytecode = True


class TestCase(utils.test_case.TestCaseAbs):
    def __init__(self):
        # Add new input files here as needed.
        #
        # The first input should also be the default --driver_input passed from
        # CMake, since the harness validates that path before Prepare() runs.
        self.cases = [
            {
                "name": "GMG, No Flux Correction, AMR, no base block coarsening, D_all=1",
                "input": "parthinput.poisson",
                "max_final_error": 1.0e-12,
                "min_final_error": 0.0,
            },
            {
                "name": "GMG, Flux Correction, AMR, base block coarsening, D_all=1",
                "input": "parthinput.poisson",
                "args": [
                    "parthenon/mesh/base_block_coarsenings=1",
                    "poisson/flux_correct=true",
                ],
                "max_final_error": 1.0e-12,
                "min_final_error": 0.0,
            },
            {
                "name": "MG-BiCSTAB, Flux Correction, AMR, base block coarsening, D_all=1",
                "input": "parthinput.poisson",
                "args": [
                    "parthenon/mesh/base_block_coarsenings=1",
                    "poisson/flux_correct=true",
                    "poisson/solver=BiCGSTAB",
                ],
                "max_final_error": 1.0e-10,
                "min_final_error": 0.0,
            },
            {
                "name": "MG-BiCSTAB, Flux Correction, AMR, base block coarsening, D_interior=1e5",
                "input": "parthinput.poisson",
                "args": [
                    "parthenon/mesh/base_block_coarsenings=1",
                    "poisson/flux_correct=true",
                    "poisson/solver=BiCGSTAB",
                    "poisson/interior_D=1e5",
                ],
                "max_final_error": 1.0e-10,
                "min_final_error": 0.0,
            },
        ]

    def Prepare(self, parameters, step):
        case = self.cases[step - 1]
        parameters.driver_input_path = os.path.join(parameters.test_path, case["input"])
        parameters.driver_cmd_line_args = list(case.get("args", []))
        return parameters

    def Analyse(self, parameters):
        success = True

        cycle_line_re = re.compile(
            r"^\s*\d+\s+[0-9.eE+\-]+(?:\s+[0-9.eE+\-]+)?\s*$", re.MULTILINE
        )
        final_error_re = re.compile(r"Final rms error:\s*([0-9.eE+\-]+)")

        for case, stdout in zip(self.cases, parameters.stdouts):
            case_success = True
            text = stdout.decode("utf-8", errors="replace")

            # Check that at least one cycle / iteration line was printed.
            if cycle_line_re.search(text) is None:
                print(f"[{case['name']}] No solver cycle output found.")
                case_success = False

            # Check that final rms error was reported.
            m = final_error_re.search(text)
            if m is None:
                print(f"[{case['name']}] Could not find 'Final rms error' in output.")
                case_success = False
            else:
                final_error = float(m.group(1))

                if final_error > case["max_final_error"]:
                    print(
                        f"[{case['name']}] Final rms error too large: "
                        f"{final_error:.6e} > {case['max_final_error']:.6e}"
                    )
                    case_success = False

                if final_error <= case["min_final_error"]:
                    print(
                        f"[{case['name']}] Final rms error suspiciously small: "
                        f"{final_error:.6e} <= {case['min_final_error']:.6e}"
                    )
                    case_success = False

                if case_success:
                    print(f"[{case['name']}] PASS: final rms error = {final_error:.6e}")

            success = success and case_success

        return success
