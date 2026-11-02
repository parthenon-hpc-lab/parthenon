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

# This file was made in part with generative AI.
import numpy as np
import sys
import utils.test_case

sys.dont_write_bytecode = True

FIELDS = ["lukewarm.x", "lukewarm.y", "lukewarm.z", "tracer.time", "tracer.velocity"]


# Restarts from an openPMD file, which holds tracers with a scalar and a vector field. An
# openPMD restart cannot hold a field whose name has a '.', so this problem's only mesh
# field is derived, which restarts do not write. Its tracer fields have a '.' too, so this
# test is not run until openPMD files can hold them (see tst/regression/CMakeLists.txt),
# and the record names expected below may change then.
class TestCase(utils.test_case.TestCaseAbs):
    def Prepare(self, parameters, step):
        parameters.coverage_status = "both"
        if step == 1:
            parameters.driver_cmd_line_args = [
                "parthenon/job/problem_id=gold",
                "parthenon/output1/file_type=openpmd",
                "parthenon/output1/output_type=restart",
                "parthenon/output1/dt=0.25",
                "parthenon/output2/file_type=hdf5",
                "parthenon/output2/dt=0.25",
                "parthenon/output2/lukewarms=tracers",
                "parthenon/output2/lukewarm_variables=tracer.time,tracer.velocity",
            ]
        elif step == 2:
            parameters.driver_cmd_line_args = [
                "-r",
                "gold.out1.00002.bp",
                "parthenon/job/problem_id=silver",
                # The restart runs on MeshData of a different size than the run it
                # restarts
                "parthenon/mesh/pack_size=2",
            ]
        elif step == 3:
            # A lukewarm with no particles must still be written, and read back
            parameters.driver_cmd_line_args = [
                "parthenon/job/problem_id=empty",
                "parthenon/time/tlim=0.5",
                "parthenon/output1/file_type=openpmd",
                "parthenon/output1/output_type=restart",
                "parthenon/output1/dt=0.25",
                "Tracers/num_tracers_per_block=0",
            ]
        else:
            parameters.driver_cmd_line_args = [
                "-r",
                "empty.out1.00001.bp",
                "parthenon/job/problem_id=empty_restarted",
            ]
        return parameters

    def Analyse(self, parameters):
        try:
            import h5py
            import openpmd_api as opmd
        except ModuleNotFoundError:
            print("Couldn't find required h5py and openpmd_api modules.")
            return False
        success = True

        def Fail(msg):
            nonlocal success
            print("TEST FAIL: " + msg)
            success = False

        # Restarting changes nothing about any tracer, though tracers in a block may come
        # in another order
        for name in ["00003", "final"]:
            files = [f"{run}.out2.{name}.phdf" for run in ["gold", "silver"]]
            data = [h5py.File(f, "r") for f in files]
            times = [d["Info"].attrs["Time"] for d in data]
            if times[0] != times[1]:
                Fail(f"{files} are at times {times}")
            ids = [d["tracers/LukewarmVars/lukewarm.id"][()] for d in data]
            order = [np.argsort(i) for i in ids]
            if (
                len(ids[0]) != len(ids[1])
                or not (ids[0][order[0]] == ids[1][order[1]]).all()
            ):
                Fail(f"{files} hold different tracers")
                continue
            for field in FIELDS:
                a, b = [
                    d["tracers/LukewarmVars/" + field][()][..., o]
                    for d, o in zip(data, order)
                ]
                if a.shape != b.shape or not (a == b).all():
                    Fail(f"{files} differ in {field}")
            for d in data:
                d.close()

        # The openPMD files name records as the standard asks
        records = {
            "id": [opmd.Record_Component.SCALAR],
            "position": ["x", "y", "z"],
            "positionOffset": ["x", "y", "z"],
            "tracer_time": [opmd.Record_Component.SCALAR],
            "tracer_velocity": ["0", "1", "2"],
        }
        for run in ["gold", "silver"]:
            for pattern, index in [("out1.%T.bp", 3), ("out1.final.%T.bp", 4)]:
                filename = f"{run}.{pattern}"
                series = opmd.Series(filename, opmd.Access.read_only)
                species = series.iterations[index].particles["tracers"]
                found = {r: sorted(species[r]) for r in species}
                if found != {r: sorted(c) for r, c in records.items()}:
                    Fail(f"{filename} has records {found}")
                ids = species["id"][opmd.Record_Component.SCALAR].load_chunk()
                series.flush()
                if sum(species.get_attribute("counts")) != len(ids):
                    Fail(f"{filename} counts do not add up")
                if species.get_attribute("id_bound") <= ids.max():
                    Fail(f"{filename} has ids that are not below its id_bound")
                series.close()

        # openPMD drops a species without records on read, so an empty lukewarm is
        # written with zero-length ones
        for run in ["empty", "empty_restarted"]:
            filename = f"{run}.out1.final.%T.bp"
            series = opmd.Series(filename, opmd.Access.read_only)
            particles = series.iterations[2].particles
            if "tracers" not in particles:
                Fail(f"{filename} has no tracers")
            else:
                species = particles["tracers"]
                found = {r: sorted(species[r]) for r in species}
                if found != {r: sorted(c) for r, c in records.items()}:
                    Fail(f"{filename} has records {found}")
                extent = species["id"][opmd.Record_Component.SCALAR].shape
                if list(extent) != [0] or sum(species.get_attribute("counts")) != 0:
                    Fail(f"{filename} should hold no tracers, but has extent {extent}")
                if "id_bound" not in species.attributes:
                    Fail(f"{filename} has no id_bound")
            series.close()
        return success
