# ========================================================================================
# Parthenon performance portable AMR framework
# Copyright(C) 2024-2026 The Parthenon collaboration
# Licensed under the 3-clause BSD License, see LICENSE file for details
# ========================================================================================
# This file was made in part with generative AI.

# Modules
import sys
import utils.test_case

# To prevent littering up imported folders with .pyc files or __pycache__ folder
sys.dont_write_bytecode = True


# Smoke test that sparse_advection OpenPMD output is conformant.
# Opens the resulting .bp series with openpmd_api and verifies:
#   1. The series opens without error (non-conformant files cause openpmd_api to throw).
#   2. Every mesh record component has a valid (non-empty) shape.
#      An empty shape indicates a ghost record — a mesh record whose attributes were
#      written immediately but whose ADIOS2 variable was never populated (no storeChunk
#      call), which violates the OpenPMD standard.
class TestCase(utils.test_case.TestCaseAbs):
    def Prepare(self, parameters, step):
        parameters.coverage_status = "both"
        parameters.driver_cmd_line_args = [
            "parthenon/output0/file_type=openpmd",
            "parthenon/output0/variables=sparse,shape_shift,dense_A,dense_B",
            "sparse_advection/restart_test=true",
            "parthenon/time/tlim=0.5",
        ]
        return parameters

    def Analyse(self, parameters):
        try:
            import openpmd_api as opmd
        except ModuleNotFoundError:
            print("Couldn't find required openpmd_api module.")
            return False

        try:
            series = opmd.Series("sparse.out0.%T.bp", opmd.Access.read_only)
            if len(series.iterations) == 0:
                print("No iterations in OpenPMD output.")
                return False

            for it_idx in series.iterations:
                it = series.iterations[it_idx]
                it.open()
                if len(it.meshes) == 0:
                    print(f"No mesh records in iteration {it_idx}.")
                    return False
                for mesh_name, mesh in it.meshes.items():
                    for comp_name, comp in mesh.items():
                        if not comp.shape:
                            print(
                                f"Ghost record detected: '{mesh_name}/{comp_name}'"
                                f" in iteration {it_idx} has no dataset."
                            )
                            return False

            series.close()
        except Exception as e:
            print(f"Failed to open or read OpenPMD output: {e}")
            return False

        return True
