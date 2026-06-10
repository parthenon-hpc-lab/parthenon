# ========================================================================================
# Parthenon performance portable AMR framework
# Copyright(C) 2026 The Parthenon collaboration
# Licensed under the 3-clause BSD License, see LICENSE file for details
# ========================================================================================

import re
import sys
import utils.test_case

sys.dont_write_bytecode = True

# Maximum tolerated absolute error for a forward+inverse FFT round-trip on a
# 16^3 grid.  Double-precision FFTs accumulate O(N log N) rounding errors, so
# for N=16 the expected residual is well below 1e-10.
ERROR_TOLERANCE = 1e-10


class TestCase(utils.test_case.TestCaseAbs):
    def Prepare(self, parameters, step):
        parameters.coverage_status = "both"
        return parameters

    def Analyse(self, parameters):
        if not parameters.stdouts:
            print("ERROR: no stdout captured from driver")
            return False

        output = parameters.stdouts[0].decode()

        match = re.search(
            r"Max relative error after FFT round-trip:\s+([\d.eE+\-]+)", output
        )
        if match is None:
            print("ERROR: could not find 'Max relative error after FFT round-trip' in output")
            print("Driver output was:")
            print(output)
            return False

        max_error = float(match.group(1))
        print("Max FFT round-trip error: {:.3e}  (tolerance: {:.3e})".format(
            max_error, ERROR_TOLERANCE
        ))

        if max_error > ERROR_TOLERANCE:
            print("ERROR: FFT round-trip error {:.3e} exceeds tolerance {:.3e}".format(
                max_error, ERROR_TOLERANCE
            ))
            return False

        return True
