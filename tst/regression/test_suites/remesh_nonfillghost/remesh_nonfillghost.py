# ========================================================================================
# Parthenon performance portable AMR framework
# Copyright(C) 2026 The Parthenon collaboration
# Licensed under the 3-clause BSD License, see LICENSE file for details
# ========================================================================================

# Modules
import sys
import numpy as np
import utils.test_case

# To prevent littering up imported folders with .pyc files or __pycache_ folder
sys.dont_write_bytecode = True


class TestCase(utils.test_case.TestCaseAbs):
    def Prepare(self, parameters, step):
        return parameters

    def Analyse(self, parameters):
        fname = "remesh_nonfillghost.out0.hst"
        with open(fname) as f:
            header = [l for l in f if l.startswith("#")][-1]
        columns = {}
        for entry in header[1:].split():
            idx, name = entry.split("=")
            columns[name] = int(idx.strip("[]")) - 1
        data = np.loadtxt(fname)

        nblocks = data[:, columns["nbtotal"]]
        if np.all(nblocks == nblocks[0]):
            print("The mesh never changed, so this test does not test remeshing.")
            return False

        passed = True
        for name in ("remesh_tracer_cc_err", "remesh_tracer_face_err"):
            err = np.max(data[:, columns[name]])
            print(f"{name}: max error {err:.3e}")
            if err > 1.0e-12:
                print(f"  {name} was not preserved across remeshing")
                passed = False
        return passed
