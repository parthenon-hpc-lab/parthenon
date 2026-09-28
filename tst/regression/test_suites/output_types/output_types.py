# Parthenon performance portable AMR framework
# Copyright(C) 2026 The Parthenon collaboration
# Licensed under the 3-clause BSD License, see LICENSE file for details

from pathlib import Path
import shutil

import h5py
import numpy as np
import utils.test_case


class TestCase(utils.test_case.TestCaseAbs):
    def Prepare(self, parameters, step):
        parameters.coverage_status = "both"
        self.expected_error = step >= 7
        if step == 1:
            self.failed_steps = 0
            parameters.driver_cmd_line_args = []
        elif step in (2, 3, 4, 5):
            source = "types.restart.00000.phdf"
            if step == 3:
                # Emulate a legacy checkpoint with no output-type metadata.
                source = "legacy.rhdf"
                shutil.copyfile("types.restart.00000.phdf", source)
                with h5py.File(source, "r+") as f:
                    del f["Info"].attrs["OutputType"]
            elif step == 4:
                source = "unmarked.phdf"
                shutil.copyfile("types.restart.00000.phdf", source)
                with h5py.File(source, "r+") as f:
                    del f["Info"].attrs["OutputType"]
            elif step == 5:
                # Saved input contains a restart block, but this file is data output.
                source = "types.out1.00000.phdf"
            parameters.driver_cmd_line_args = [
                "-r",
                source,
                f"parthenon/job/problem_id=resume{step}",
            ]
        elif step == 6:
            # Exercise legacy aliases and an explicit filename ID.
            input_file = Path(parameters.output_path) / "legacy.in"
            text = Path(parameters.driver_input_path).read_text()
            text = text.replace(
                "file_type = hdf5\noutput_type = restart", "file_type = rst"
            )
            text = text.replace(
                "file_type = hdf5\noutput_type = core", "file_type = corehdf5"
            )
            input_file.write_text(text)
            parameters.driver_cmd_line_args = [
                "-i",
                str(input_file),
                "parthenon/job/problem_id=aliases",
                "parthenon/output0/id=checkpoint",
            ]
        elif step == 7:
            parameters.driver_cmd_line_args = ["parthenon/output1/output_type=restart"]
        elif step == 8:
            parameters.driver_cmd_line_args = ["parthenon/output1/file_type=rst"]
        elif step == 9:
            parameters.driver_cmd_line_args = ["parthenon/output1/output_type=core"]
        elif step == 10:
            parameters.driver_cmd_line_args = ["parthenon/output1/output_type=x1slice"]
        return parameters

    def ErrorOnNonZeroReturnCode(self, parameters, returncode):
        if self.expected_error:
            self.failed_steps += 1
            return False
        return True

    def Analyse(self, parameters):
        assert self.failed_steps == 4, "Invalid output configurations must fail"
        warning = "not written with output_type=restart"
        for step, output in enumerate(parameters.stdouts[:6], 1):
            assert (warning in output.decode()) == (step in (4, 5))
        for output in parameters.stdouts[6:8]:
            assert "More than one restart output block" in output.decode()
        assert "More than one corehdf5 output block" in parameters.stdouts[8].decode()
        assert "restart data core" in parameters.stdouts[9].decode()

        for name, mode in [("restart", "restart"), ("out1", "data"), ("out2", "core")]:
            with h5py.File(f"types.{name}.00000.phdf", "r") as f:
                value = np.asarray(f["Info"].attrs["OutputType"]).item()
                if isinstance(value, bytes):
                    value = value.decode()
                assert value == mode
                assert "advected" in f
                # The core dump includes derived fields, while restart/data do not.
                assert ("one_minus_advected" in f) == (mode == "core")
        assert Path("aliases.checkpoint.00000.phdf").is_file()
        assert Path("aliases.out2.00000.phdf").is_file()
        assert not list(Path(".").glob("*.chdf"))
        assert sorted(p.name for p in Path(".").glob("*.rhdf")) == ["legacy.rhdf"]

        with h5py.File("types.restart.final.phdf", "r") as gold:
            for step in (2, 3, 4, 5):
                with h5py.File(f"resume{step}.restart.final.phdf", "r") as resumed:
                    np.testing.assert_array_equal(
                        gold["advected"][:], resumed["advected"][:]
                    )
        return True
