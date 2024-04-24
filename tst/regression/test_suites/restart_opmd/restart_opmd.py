# ========================================================================================
# Parthenon performance portable AMR framework
# Copyright(C) 2024 The Parthenon collaboration
# Licensed under the 3-clause BSD License, see LICENSE file for details
# ========================================================================================

# Modules
import sys
import utils.test_case


# To prevent littering up imported folders with .pyc files or __pycache_ folder
sys.dont_write_bytecode = True


class TestCase(utils.test_case.TestCaseAbs):
    def Prepare(self, parameters, step):
        # enable coverage testing on pass where restart
        # files are both read and written
        parameters.coverage_status = "both"

        # run baseline (to the very end)
        if step == 1:
            parameters.driver_cmd_line_args = ["parthenon/job/problem_id=gold"]
        # restart from an early snapshot
        elif step == 2:
            parameters.driver_cmd_line_args = [
                "-r",
                "gold.out1.00001.bp",
                "-i",
                f"{parameters.parthenon_path}/tst/regression/test_suites/restart_opmd/parthinput_override.restart",
            ]

        return parameters

    def Analyse(self, parameters):
        try:
            import openpmd_api as ompd
        except ModuleNotFoundError:
            print("Couldn't find required openpmd_api module to compare test results.")
            return False
        success = True

        def compare_attributes(series_a, series_b):
            all_equal = True
            for attr in series_a.attributes:
                if series_b.contains_attribute(attr):
                    attr_a = series_a.get_attribute(attr)
                    attr_b = series_b.get_attribute(attr)
                    if attr_a != attr_b:
                        print(f"Mismatch in attribute '{attr}'. "
                              f"'{attr_a}' versus '{attr_b}'\n"
                              )
                        all_equal = False
                else:
                    print(f"Missing attribute '{attr}' in second file.")
                    all_equal = False
            return all_equal
                    



        def compare_files(name):
            series_gold = opmd.Series("gold.out1.%T.bp/", opmd.Access.read_only)
            series_silver = opmd.Series("silver.out1.%T.bp/", opmd.Access.read_only)
            delta = compare(
                [
                    "gold.out0.%s.rhdf" % name,
                    "silver.out0.%s.rhdf" % name,
                ],
                one=True,
            )

            if delta != 0:
                print(
                    "ERROR: Found difference between gold and silver output '%s'."
                    % name
                )
                return False

            return True

        # comapre a few files throughout the simulations
        success &= compare_files("00002")
        success &= compare_files("00005")
        success &= compare_files("00009")
        success &= compare_files("final")

        found_line = False
        for line in parameters.stdouts[1].decode("utf-8").split("\n"):
            if "Terminating on wall-time limit" in line:
                found_line = True
        if not found_line:
            print("ERROR: wall-time limit based termination not triggered.")
            success = False

        return success
