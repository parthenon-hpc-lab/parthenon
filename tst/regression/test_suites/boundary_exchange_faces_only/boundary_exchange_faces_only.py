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
# This file was created in part with generative AI.

# Modules
import os
import sys

import utils.test_case

# To prevent littering up imported folders with .pyc files or __pycache_ folder
sys.dont_write_bytecode = True


class TestCase(utils.test_case.TestCaseAbs):
    def __init__(self):
        # The first input should also be the default --driver_input passed from CMake
        self.cases = [
            {"name": "forest", "input": "parthinput.forest"},
            {
                "name": "forest, coalesced comms",
                "input": "parthinput.forest",
                "args": ["parthenon/mesh/do_coalesced_comms=true"],
            },
            {
                "name": "2D periodic mesh",
                "input": "parthinput.mesh",
                "args": [
                    "parthenon/mesh/nx3=1",
                    "parthenon/meshblock/nx3=1",
                    "parthenon/mesh/ix1_bc=periodic",
                    "parthenon/mesh/ox1_bc=periodic",
                    "parthenon/mesh/ix2_bc=periodic",
                    "parthenon/mesh/ox2_bc=periodic",
                ],
            },
            {"name": "3D mesh", "input": "parthinput.mesh"},
            {
                "name": "3D periodic mesh, coalesced comms",
                "input": "parthinput.mesh",
                "args": [
                    "parthenon/mesh/do_coalesced_comms=true",
                    "parthenon/mesh/ix1_bc=periodic",
                    "parthenon/mesh/ox1_bc=periodic",
                    "parthenon/mesh/ix2_bc=periodic",
                    "parthenon/mesh/ox2_bc=periodic",
                    "parthenon/mesh/ix3_bc=periodic",
                    "parthenon/mesh/ox3_bc=periodic",
                ],
            },
        ]

    def Prepare(self, parameters, step):
        case = self.cases[step - 1]
        parameters.driver_input_path = os.path.join(parameters.test_path, case["input"])
        parameters.driver_cmd_line_args = list(case.get("args", []))
        return parameters

    def Analyse(self, parameters):
        success = True
        for case, stdout in zip(self.cases, parameters.stdouts):
            text = stdout.decode("utf-8", errors="replace")
            case_success = (
                "faces_only check PASSED" in text and "send/recv mismatch" not in text
            )
            for line in text.splitlines():
                if line.startswith("faces_only"):
                    print(f"[{case['name']}] {line}")
            print(f"[{case['name']}] {'PASS' if case_success else 'FAIL'}")
            success = success and case_success
        return success
