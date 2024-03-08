# Athena++ astrophysical MHD code
# Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
# Licensed under the 3-clause BSD License, see LICENSE file for details
# ========================================================================================
# (C) (or copyright) 2020-2021. Triad National Security, LLC. All rights reserved.
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

import os
from shutil import rmtree
import subprocess
from subprocess import PIPE
import sys
from shutil import which


class Parameters:
    driver_path = ""
    driver_input_path = ""
    test_path = ""
    output_path = ""
    parthenon_path = ""
    mpi_cmd = ""
    num_ranks = 1
    mpi_opts = ""
    driver_cmd_line_args = []
    stdouts = []
    kokkos_args = []
    # Options
    # only-regression - do not run when coverage is enabled
    # both - run regardless of whether coverage is enabled or not
    # only-coverage - only run if coverage has been enabled
    coverage_status = "only-regression"


class TestCaseAbs:
    def Prepare(parameters):
        raise NotImplementedError("Every TestCase must initialse a Prepare method")
        return parameters

    def Analyse(parameters):
        raise NotImplementedError("Every TestCase must initialse an Analyse method")


class TestManager:
    def __init__(self, run_test_path, **kwargs):

        self.__run_coverage = kwargs.pop("coverage")
        self.parameters = Parameters()
        self.__run_test_py_path = run_test_path
        self.__regression_test_suite_path = os.path.join(
            self.__run_test_py_path, "test_suites"
        )
        test_dir = kwargs.pop("test_dir")
        parthenon_driver = kwargs.pop("driver")
        parthenon_driver_input = kwargs.pop("driver_input")
        self.parameters.kokkos_args = " ".join(kwargs.pop("kokkos_args")).split()
        mpi_executable = kwargs.pop("mpirun")

        self.__initial_working_dir = os.getcwd()

        test_path = self.__checkAndGetRegressionTestFolder(test_dir[0])

        test_base_name = os.path.split(test_path)[1]
        self.test = os.path.basename(os.path.normpath(test_path))

        self.__checkRegressionTestScript(test_dir[0], test_base_name)
        self.__checkDriverPath(parthenon_driver[0])
        self.__checkDriverInputPath(parthenon_driver_input[0])
        self.__checkMPIExecutable(mpi_executable)

        driver_path = os.path.abspath(parthenon_driver[0])
        driver_input_path = os.path.abspath(parthenon_driver_input[0])

        output_path = kwargs.pop("output_dir")
        if output_path == "":
            output_path = os.path.abspath(test_path + "/output")
        else:
            output_path = os.path.abspath(output_path)

        try:
            parthenon_path = os.path.realpath(__file__)
            idx = parthenon_path.rindex("/parthenon/")
            self.parameters.parthenon_path = os.path.join(
                parthenon_path[:idx], "parthenon"
            )
        except ValueError:
            baseDir = os.path.dirname(__file__)
            self.parameters.parthenon_path = os.path.abspath(baseDir + "/../../../")

        self.__test_module = "test_suites." + test_base_name + "." + test_base_name

        output_msg = "Using:\n"
        output_msg += "driver at:       " + driver_path + "\n"
        output_msg += "driver input at: " + driver_input_path + "\n"
        output_msg += "test folder:     " + test_path + "\n"
        output_msg += "output sent to:  " + output_path + "\n"
        print(output_msg)
        sys.stdout.flush()

        self.parameters.driver_path = driver_path
        self.parameters.driver_input_path = driver_input_path
        self.parameters.output_path = output_path
        self.parameters.test_path = test_path
        self.parameters.mpi_cmd = mpi_executable
        self.parameters.mpi_ranks_flag = kwargs.pop("mpirun_ranks_flag")
        self.parameters.num_ranks = int(kwargs.pop("mpirun_ranks_num"))
        self.parameters.mpi_opts = kwargs.pop("mpirun_opts")
        self.parameters.sparse_disabled = kwargs.pop("sparse_disabled")

        module_root_path = os.path.join(test_path, "..", "..")
        if module_root_path not in sys.path:
            sys.path.insert(0, module_root_path)
        module = __import__(
            self.__test_module, globals(), locals(), fromlist=["TestCase"]
        )
        my_TestCase = getattr(module, "TestCase")
        self.test_case = my_TestCase()

        if not issubclass(my_TestCase, TestCaseAbs):
            raise TestManagerError("TestCase is not a child of TestCaseAbs")

    def __checkAndGetRegressionTestFolder(self, test_dir):
        if not os.path.isdir(test_dir):
            if not os.path.isdir(os.path.join("test_suites", test_dir)):
                error_msg = "Regression test folder is unknown: " + test_dir + "\n"
                error_msg += "looked in:\n"
                error_msg += "  tst/regression/test_suites/" + test_dir + "\n"
                error_msg += "  " + test_dir + "\n"
                error_msg += "Each regression test must have a folder in "
                error_msg += "tst/regression/test_suites.\n"
                error_msg += "Known tests folders are:"
                known_test_folders = os.listdir(self.__regression_test_suite_path)
                for folder in known_test_folders:
                    error_msg += "\n  " + folder

                raise TestManagerError(error_msg)
            else:
                return os.path.abspath(os.path.join("test_suites", test_dir))
        else:
            return os.path.abspath(test_dir)

    def __checkRegressionTestScript(self, test_dir, test_base_name):
        python_test_script = os.path.join(test_dir, test_base_name + ".py")
        if not os.path.isfile(python_test_script):
            error_msg = "Missing regression test file "
            error_msg += python_test_script
            error_msg += "\nEach test folder must have a python script with the same name as the "
            error_msg += "regression test folder."
            raise TestManagerError(error_msg)

    def __checkDriverPath(self, parthenon_driver):
        if not os.path.isfile(parthenon_driver):
            raise TestManagerError("Unable to locate driver " + parthenon_driver)

    def __checkDriverInputPath(self, parthenon_driver_input):
        if not os.path.isfile(parthenon_driver_input):
            raise TestManagerError(
                "Unable to locate driver input file " + parthenon_driver_input
            )

    def __checkMPIExecutable(self, mpi_executable):

        if not mpi_executable:
            return

        mpi_exec = mpi_executable[0]

        if which(mpi_exec) is None:
            error_msg = "mpi executable path provided, but no file found: "
            error_msg += mpi_exec
            raise TestManagerError(error_msg)

    def MakeOutputFolder(self):
        if os.path.isdir(self.parameters.output_path):
            try:
                rmtree(self.parameters.output_path)
            except OSError as e:
                print("Error: %s - %s." % (e.filename, e.strerror))

        if not os.path.isdir(self.parameters.output_path):
            os.makedirs(self.parameters.output_path)

        os.chdir(self.parameters.output_path)

    def Prepare(self, step):
        print("*****************************************************************")
        print("Preparing Test Case Step %d" % step)
        print("*****************************************************************\n")
        sys.stdout.flush()
        self.parameters = self.test_case.Prepare(self.parameters, step)

    def Run(self):

        run_command = []
        if self.parameters.mpi_cmd != "":
            run_command.extend(self.parameters.mpi_cmd)
        if self.parameters.mpi_ranks_flag is not None:
            run_command.append(self.parameters.mpi_ranks_flag)
            run_command.append(str(self.parameters.num_ranks))
        for opt in self.parameters.mpi_opts:
            run_command.extend(opt.split())
        run_command.append(self.parameters.driver_path)
        if not "-r" in self.parameters.driver_cmd_line_args:
            run_command.append("-i")
            run_command.append(self.parameters.driver_input_path)
        for arg in self.parameters.driver_cmd_line_args:
            run_command.append(arg)
        for arg in self.parameters.kokkos_args:
            run_command.append(arg)

        if self.__run_coverage and self.parameters.coverage_status != "only-regression":
            print("*****************************************************************")
            print("Running Driver with Coverage")
            print("*****************************************************************\n")
        elif (
            not self.__run_coverage
            and self.parameters.coverage_status != "only-coverage"
        ):
            print("*****************************************************************")
            print("Running Driver")
            print("*****************************************************************\n")
        elif (
            self.__run_coverage and self.parameters.coverage_status == "only-regression"
        ):
            print("*****************************************************************")
            print("Test Case Ignored for Calculating Coverage")
            print("*****************************************************************\n")
            return
        else:
            return

        print("Command to execute driver")
        print(" ".join(run_command))
        sys.stdout.flush()
        try:
            proc = subprocess.run(run_command, check=True)
            self.parameters.stdouts.append(proc.stdout)
        except subprocess.CalledProcessError as err:
            print("\n*****************************************************************")
            print("Subprocess error message")
            print("*****************************************************************\n")
            print(str(repr(err.output)).replace("\\n", os.linesep))
            print("\n*****************************************************************")
            print("Error detected while running subprocess command")
            print("*****************************************************************\n")
            raise TestManagerError(
                "\nReturn code {0} from command '{1}'".format(
                    err.returncode, " ".join(err.cmd)
                )
            )
        # Reset parameters
        self.parameters.coverage_status = "only-regression"

    def Analyse(self):

        test_pass = False
        if self.__run_coverage:
            print("*****************************************************************")
            print("Running with Coverage, Analysis Section Ignored")
            print("*****************************************************************\n")
            return True

        print("Running with coverage")
        print(self.__run_coverage)
        print("*****************************************************************")
        print("Analysing Driver Output")
        print("*****************************************************************\n")
        sys.stdout.flush()
        test_pass = self.test_case.Analyse(self.parameters)

        return test_pass


# Exception for unexpected behavior by individual tests
class TestManagerError(RuntimeError):
    pass
