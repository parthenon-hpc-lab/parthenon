# ========================================================================================
# Parthenon performance portable AMR framework
# Copyright(C) 2026 The Parthenon collaboration
# Licensed under the 3-clause BSD License, see LICENSE file for details
# ========================================================================================
# This file was made in part with generative AI.

# Modules
import pathlib
import sys
import numpy as np
import utils.test_case

# To prevent littering up imported folders with .pyc files or __pycache__ folder
sys.dont_write_bytecode = True


# Checks that a streamed openPMD output (stream_engine = bp5) contains the same data as
# the file based output with the same cadence (read linearly as a live viewer would) and
# that an output with on_request = true is only written once for a single request.
class TestCase(utils.test_case.TestCaseAbs):
    def Prepare(self, parameters, step):
        pathlib.Path("advection.request.request").touch()
        return parameters

    def Analyse(self, parameters):
        sys.path.insert(
            1,
            parameters.parthenon_path
            + "/scripts/python/packages/parthenon_tools/parthenon_tools",
        )
        try:
            import openpmd_api as opmd
            from opmd_live import composite
        except ModuleNotFoundError:
            print("Couldn't find required openpmd_api or opmd_live module.")
            return False

        def read(series, iterations):
            data = {}
            for idx, it in iterations:
                # with AMR, records of finer levels only cover the refined regions
                chunks = {}
                for name, mesh in it.meshes.items():
                    rc = mesh[opmd.Mesh_Record_Component.SCALAR]
                    for c in rc.available_chunks():
                        key = (name, tuple(c.offset), tuple(c.extent))
                        chunks[key] = rc.load_chunk(c.offset, c.extent)
                series.flush()
                data[idx] = (it.time, chunks)
                it.close()
            return data

        files = opmd.Series("advection.out0.%05T.bp", opmd.Access.read_only)
        ref = read(files, files.iterations.items())
        stream = opmd.Series("advection.stream.bp", opmd.Access.read_linear)
        streamed = read(stream, stream.snapshots().items())
        requested = opmd.Series("advection.request.bp", opmd.Access.read_linear)
        num_requested = len(read(requested, requested.snapshots().items()))
        coarse = opmd.Series("advection.coarse.bp", opmd.Access.read_linear)
        coarsened = read(coarse, coarse.snapshots().items())

        success = True
        # the final output (only written to file with the 'final' label) is not streamed
        if sorted(streamed) != sorted(ref) or len(streamed) < 4:
            print(f"Mismatch in iterations: {sorted(streamed)} vs {sorted(ref)}")
            success = False
        for idx, (time, values) in streamed.items():
            if idx not in ref:
                continue
            ref_time, ref_values = ref[idx]
            if (
                time != ref_time
                or values.keys() != ref_values.keys()
                or not all(np.array_equal(v, ref_values[k]) for k, v in values.items())
            ):
                print(f"Data of streamed iteration {idx} does not match file output.")
                success = False
        # coarsened output contains every 2nd cell (starting with the first one per block)
        for idx, (time, values) in coarsened.items():
            ref_values = {
                (n, tuple(o // 2 for o in off), tuple(e // 2 for e in ext)): v[::2, ::2]
                for (n, off, ext), v in ref[idx][1].items()
            }
            if (
                values.keys() != ref_values.keys()
                or any(v.dtype != np.float32 for v in values.values())
                or not all(
                    np.array_equal(v, ref_values[k].astype(np.float32))
                    for k, v in values.items()
                )
            ):
                print(f"Data of coarsened iteration {idx} does not match file output.")
                success = False
        if sorted(coarsened) != sorted(streamed):
            print(f"Mismatch in coarsened iterations: {sorted(coarsened)}")
            success = False
        # combining all levels on a uniform grid of any level conserves the mean
        files = opmd.Series("advection.out0.%05T.bp", opmd.Access.read_only)
        it = files.iterations[max(ref)]
        means = [np.mean(composite(files, it, "advected", l)[0]) for l in range(-2, 3)]
        if not np.all(np.isfinite(means)) or not np.allclose(
            means, means[0], rtol=1e-5
        ):
            print(f"Composite data not conservative: {means}")
            success = False
        if num_requested != 1:
            print(f"Expected one requested output but found {num_requested}.")
            success = False
        if pathlib.Path("advection.request.request").exists():
            print("Request file was not removed.")
            success = False
        return success
