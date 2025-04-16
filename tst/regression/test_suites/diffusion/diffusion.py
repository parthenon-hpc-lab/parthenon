# ========================================================================================
# Parthenon performance portable AMR framework
# Copyright(C) 2020 The Parthenon collaboration
# Licensed under the 3-clause BSD License, see LICENSE file for details
# ========================================================================================
# (C) (or copyright) 2021-2025. Triad National Security, LLC. All rights reserved.
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

# Modules
import sys
from typing import Union  # python < 3.10
import utils.test_case

import numpy as np

# To prevent littering up imported folders with .pyc files or __pycache_ folder
sys.dont_write_bytecode = True


def analytic_solution(
    x: np.ndarray, t: float, D: Union[float, np.ndarray], t0: float
) -> np.ndarray:
    """analytic solution for the diffusion of a constant coefficient Gaussian"""
    xc = 0.0
    return np.sqrt(t0 / (t + t0)) * np.exp(-0.25 * (x - xc) ** 2 / (D * (t + t0)))


class TestCase(utils.test_case.TestCaseAbs):
    def Prepare(self, parameters, step):
        return parameters

    def Analyse(self, parameters):
        sys.path.insert(
            1,
            parameters.parthenon_path
            + "/scripts/python/packages/parthenon_tools/parthenon_tools",
        )
        try:
            from phdf import phdf
        except ModuleNotFoundError:
            print("Couldn't find module to open Parthenon hdf5 files.")
            return False

        data_filename = "diffusion.out0.final.phdf"
        data_file = phdf(data_filename)
        q = data_file.Get("diffusion.u", False)
        x = data_file.x
        t = data_file.Time

        NB = q.shape[0]
        t0 = data_file.Params["diffusion_package/t0"]
        D = 1.0
        error = (
            np.sum(
                np.sum(
                    np.power(q[:, 0, 0, :] - analytic_solution(x, t, D, t0), 2.0),
                    axis=1,
                )
                / np.array([len(q[b, 0, 0, :]) for b in range(NB)]),
                axis=0,
            )
            / NB
        )
        tol = 1.0e-7
        return error < tol
