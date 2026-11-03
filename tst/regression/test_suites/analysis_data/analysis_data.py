# This file was made in part with generative AI.

import sys
from pathlib import Path

import h5py
import numpy as np

import utils.test_case


sys.dont_write_bytecode = True


class TestCase(utils.test_case.TestCaseAbs):
    @staticmethod
    def _check_equal(actual, expected, description):
        if not np.array_equal(actual, expected):
            print(f"ERROR: Incorrect {description}")
            return False
        return True

    @staticmethod
    def _minmod(left_difference, right_difference):
        if left_difference * right_difference <= 0.0:
            return 0.0
        return np.copysign(
            min(abs(left_difference), abs(right_difference)), left_difference
        )

    def _check_ghosts(self, data, levels, locations, field, same_level_width=2):
        success = True
        order = np.argsort(locations[:, 2])
        blocks = data[order, 0, 0, 0, :]
        ordered_levels = levels[order]

        if not np.array_equal(ordered_levels, np.array([0, 1, 1, 1, 1, 0])):
            print(f"ERROR: Unexpected AMR block layout for {field}")
            return False

        left_coarse = blocks[0]
        fine = blocks[1:5]
        right_coarse = blocks[5]

        success &= self._check_equal(
            left_coarse[:2],
            np.repeat(left_coarse[2], 2),
            f"left outflow ghosts for {field}",
        )
        success &= self._check_equal(
            right_coarse[6:],
            np.repeat(right_coarse[5], 2),
            f"right outflow ghosts for {field}",
        )

        for block in range(3):
            success &= self._check_equal(
                fine[block, 6 : 6 + same_level_width],
                fine[block + 1, 2 : 2 + same_level_width],
                f"right same-level ghosts for {field} block {block}",
            )
            success &= self._check_equal(
                fine[block + 1, 2 - same_level_width : 2],
                fine[block, 6 - same_level_width : 6],
                f"left same-level ghosts for {field} block {block + 1}",
            )

        fine_interior = fine[:, 2:6].reshape(-1)
        restricted = 0.5 * (fine_interior[::2] + fine_interior[1::2])
        success &= self._check_equal(
            left_coarse[6:], restricted[:2], f"coarse-from-fine ghosts for {field}"
        )
        success &= self._check_equal(
            right_coarse[:2], restricted[-2:], f"coarse-from-fine ghosts for {field}"
        )

        coarse = np.concatenate((left_coarse[2:6], restricted, right_coarse[2:6]))
        for fine_block, coarse_index, side in (
            (fine[0], 3, "left"),
            (fine[-1], 12, "right"),
        ):
            center = coarse[coarse_index]
            slope = self._minmod(
                center - coarse[coarse_index - 1],
                coarse[coarse_index + 1] - center,
            )
            expected = np.array([center - 0.25 * slope, center + 0.25 * slope])
            actual = fine_block[:2] if side == "left" else fine_block[6:]
            success &= self._check_equal(
                actual, expected, f"coarse-to-fine {side} ghosts for {field}"
            )

        return success

    def Prepare(self, parameters, step):
        if step == 1:
            parameters.driver_cmd_line_args = [
                "parthenon/job/problem_id=analysis_source"
            ]
        else:
            analysis_input = (
                f"{parameters.parthenon_path}/tst/regression/test_suites/"
                "analysis_data/parthinput.analysis"
            )
            parameters.driver_cmd_line_args = [
                "-a",
                "analysis_source.out0.final.phdf",
                "-i",
                analysis_input,
                "Advection/cfl=0.5",
                "parthenon/time/dt=0.5",
                "parthenon/time/ncycle=43",
            ]
        return parameters

    def Analyse(self, parameters):
        success = True
        output = parameters.stdouts[1].decode("utf-8")
        if "This run is a restart: 0" not in output:
            print("ERROR: Analysis mode was incorrectly reported as a restart")
            success = False
        if "cycle=44" in output:
            print("ERROR: Analysis mode entered the evolution loop")
            success = False
        if Path("analysis_result.out0.final.phdf").exists():
            print("ERROR: Analysis mode wrote an ordinary final output")
            success = False

        analysis_outputs = sorted(Path(".").glob("analysis_result.out0.*.phdf"))
        if len(analysis_outputs) != 1:
            print(
                "ERROR: Analysis mode should write exactly one numbered output, got "
                f"{analysis_outputs}"
            )
            return False

        with h5py.File("analysis_source.out0.final.phdf", "r") as source, h5py.File(
            analysis_outputs[0], "r"
        ) as result:
            boundaries = [
                value.decode() if isinstance(value, bytes) else value
                for value in result["Info"].attrs["BoundaryConditions"]
            ]
            if boundaries[:2] != ["outflow", "outflow"]:
                print(
                    "ERROR: Analysis boundary override failed: " f"got {boundaries[:2]}"
                )
                success = False
            expected_time = {"Time": 0.125, "dt": 0.5, "NCycle": 43}
            for attribute, expected in expected_time.items():
                if result["Info"].attrs[attribute] != expected:
                    print(
                        f"ERROR: Analysis parameter override for {attribute} failed: "
                        f"expected {expected}, got {result['Info'].attrs[attribute]}"
                    )
                    success = False
            levels = result["Levels"][...]
            locations = result["Locations/x"][...]
            for field in ("advected", "my_derived_var"):
                source_data = source[field][...]
                result_data = result[field][...]
                if not np.array_equal(source_data[..., 2:-2], result_data[..., 2:-2]):
                    print(f"ERROR: Analysis load changed the interior of {field}")
                    success = False
                if not np.isfinite(result_data).all():
                    print(f"ERROR: Analysis load produced non-finite values in {field}")
                    success = False
                same_level_width = 1 if field == "my_derived_var" else 2
                success &= self._check_ghosts(
                    result_data, levels, locations, field, same_level_width
                )

            for field in ("analysis_excluded_var", "bnd_flux::advected"):
                if field in result:
                    print(f"ERROR: Excluded field {field} was written by analysis")
                    success = False
            if "analysis_omitted_var" in result:
                print("ERROR: Field omitted from the dump was written by analysis")
                success = False

            output_fields = [
                value.decode() if isinstance(value, bytes) else value
                for value in result["Info"].attrs["OutputDatasetNames"]
            ]
            expected_fields = [
                "advected",
                "my_derived_var",
                "analysis_only_var",
                "analysis_only_sparse_7",
            ]
            if set(output_fields) != set(expected_fields):
                print(
                    "ERROR: Analysis output field catalog mismatch: "
                    f"expected {expected_fields}, got {output_fields}"
                )
                success = False

            analysis_only = result["analysis_only_var"][...]
            advected = result["advected"][:, 0, ...]
            if not np.array_equal(analysis_only[..., 2:-2], 2.0 * advected[..., 2:-2]):
                print("ERROR: Metadata::Analysis field was not available to analysis")
                success = False

            analysis_only_sparse = result["analysis_only_sparse_7"][...]
            if not np.array_equal(
                analysis_only_sparse[..., 2:-2], 4.0 * advected[..., 2:-2]
            ):
                print("ERROR: Sparse Metadata::Analysis field was not allocated")
                success = False

        return success
