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
# 16^3 grid.
# Error tolerance currently set for double prec data and transforms.
ERROR_TOLERANCE = 1e-14


class TestCase(utils.test_case.TestCaseAbs):
    def Prepare(self, parameters, step):
        parameters.coverage_status = "both"
        return parameters

    def Analyse(self, parameters):
        if not parameters.stdouts:
            print("ERROR: no stdout captured from driver")
            return False

        output = parameters.stdouts[0].decode()

        def parse_error(error_string):
            match = re.search(rf"{error_string}:\s+([\d.eE+\-]+)", output)
            if match is None:
                print(f"ERROR: could not find '{error_string}' in output")
                print("Driver output was:")
                print(output)
                return False

            max_error = float(match.group(1))
            print(
                f"{error_string}: {max_error:.3e}  (tolerance: {ERROR_TOLERANCE:.3e})"
            )

            if max_error > ERROR_TOLERANCE:
                print("ERROR: exceeds tolerance")
                return False
            return True

        success = True
        success &= parse_error("Max relative error after FFT round-trip")
        success &= parse_error("Error in spectrum total power")
        success &= parse_error("Error in spectrum mean")
        return success
