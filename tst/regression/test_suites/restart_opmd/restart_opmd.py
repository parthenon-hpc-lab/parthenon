# ========================================================================================
# Parthenon performance portable AMR framework
# Copyright(C) 2024 The Parthenon collaboration
# Licensed under the 3-clause BSD License, see LICENSE file for details
# ========================================================================================

# Modules
import sys
import utils.test_case
import numpy as np


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
            import openpmd_api as opmd
        except ModuleNotFoundError:
            print("Couldn't find required openpmd_api module to compare test results.")
            return False
        success = True

        def compare_attributes(series_a, series_b):
            skip_attributes = [
                "iterationFormat",  # Stores the file name format. Expected to differ.
                "WallTime",
                "InputFile",  # Is updated during runtime, e.g., startime and thus differs
            ]
            all_equal = True
            for attr in series_a.attributes:
                if series_b.contains_attribute(attr):
                    attr_a = series_a.get_attribute(attr)
                    attr_b = series_b.get_attribute(attr)
                    if attr not in skip_attributes and attr_a != attr_b:
                        print(
                            f"Mismatch in attribute '{attr}'. "
                            f"'{attr_a}' versus '{attr_b}'\n"
                        )
                        all_equal = False
                else:
                    print(f"Missing attribute '{attr}' in second file.")
                    all_equal = False
            return all_equal

        # need series in order to flush
        def compare_data(it_a, it_b, series_a, series_b):
            all_equal = True
            for mesh_name, mesh_a in it_a.meshes.items():
                if mesh_name not in it_b.meshes:
                    print(f"Missing mesh '{mesh_name}' in second file.")
                    all_equal = False
                    continue
                mesh_b = it_b.meshes[mesh_name]

                for comp_name, comp_a in mesh_a.items():
                    if comp_name not in mesh_b:
                        print(
                            f"Missing component '{comp_name}' in mesh '{mesh_name}' of second file."
                        )
                        all_equal = False
                        continue
                    comp_b = mesh_b[comp_name]
                    data_a = comp_a.load_chunk()
                    series_a.flush()
                    data_b = comp_b.load_chunk()
                    series_b.flush()

                    try:
                        np.testing.assert_array_max_ulp(data_a, data_b)
                    except AssertionError as err:
                        print(
                            f"Data of component '{comp_name}' in mesh '{mesh_name}' does not match:\n"
                            f"{err}\n"
                        )
                        all_equal = False
                        continue

            return all_equal

        def compare_files(idx_it):
            all_good = True
            series_gold = opmd.Series("gold.out1.%T.bp/", opmd.Access.read_only)
            series_silver = opmd.Series("silver.out1.%T.bp/", opmd.Access.read_only)

            # PG: yes, this is inefficient but keeps the logic simple
            all_good &= compare_attributes(series_gold, series_silver)
            all_good &= compare_attributes(series_silver, series_gold)

            it_gold = series_gold.iterations[idx_it]
            it_silver = series_silver.iterations[idx_it]
            all_good &= compare_attributes(it_gold, it_silver)
            all_good &= compare_attributes(it_silver, it_gold)

            all_good &= compare_data(it_silver, it_gold, series_silver, series_gold)
            all_good &= compare_data(it_gold, it_silver, series_gold, series_silver)

            return all_good

        # comapre a few files throughout the simulations
        success &= compare_files(1)
        success &= compare_files(2)
        success &= compare_files(3)
        success &= compare_files(4)
        # success &= compare_files("final")

        return success
