# =========================================================================================
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
# =========================================================================================

from __future__ import print_function

# ****************************************************************
# Note: reader import occurs after we fix the path at the bottom
# ****************************************************************

# **************
# other imports
import os
import sys
import numpy as np
import argparse


def Usage():
    print(
        """

    Usage: %s [-quiet] [-brief] [-all] [-one] [--tol=eps] [-ignore_metadata] file1.phdf file2.phdf

                  -all: report all diffs at all positions
                  -one: Quit after first different variable
                -brief: Only report if files are different
                        Overrides --all
                -quiet: Only report if files are different and
                        don't print any extraneous info.
             --tol=eps: set tolerance to eps.  Default 1.0e-12
      -ignore_metadata: Ignore differences in metadata

    This example takes two hdf files and compares them to see if there are
    differences in the state variables.

    Shows how to load up a structure and prints one of the structures

"""
        % os.path.basename(__file__)
    )


def processArgs():
    parser = argparse.ArgumentParser(
        description="""
    arguments for differencing script
    """
    )
    parser.add_argument(
        "-a", "-all", action="store_true", help="report all diffs at all positions"
    )
    parser.add_argument(
        "-t",
        "--tol",
        action="store",
        help="Sets tolerance for comparisons.  Default 1e-12",
    )
    parser.add_argument(
        "-o",
        "-one",
        action="store_true",
        help="Only report data for first different variable.",
    )
    parser.add_argument(
        "-b",
        "-brief",
        action="store_true",
        help="Only report if files are different.  Overrides -all",
    )
    parser.add_argument(
        "-q",
        "-quiet",
        action="store_true",
        help="Only report if files are different.  No other output. Overrides -all",
    )
    parser.add_argument(
        "-i",
        "-ignore_metadata",
        action="store_true",
        help="Ignore differences in metadata. Overrides -all",
    )
    parser.add_argument("files", nargs="*")

    return parser.parse_args()


def addPath():
    """add the vis/python directory to the pythonpath variable"""
    myPath = os.path.realpath(os.path.dirname(__file__))
    # sys.path.insert(0,myPath+'/../vis/python')
    # sys.path.insert(0,myPath+'/vis/python')


def ensure_list(x):
    return x if isinstance(x, np.ndarray) else [x]


def compare_attributes(dict0, dict1):
    keys0 = set(dict0.keys())
    keys1 = set(dict1.keys())
    union = keys0.union(keys1)
    intersect = keys0.intersection(keys1)

    # keys that only show up in one set
    diff_keys = list(keys0.symmetric_difference(keys1))

    # now compare values of keys that are in both sets
    for k in intersect:
        a = np.array(ensure_list(dict0[k]))
        b = np.array(ensure_list(dict1[k]))

        if len(a) != len(b):
            diff_keys.append(k)

        if len(a) == len(b):
            if np.any(a != b):
                diff_keys.append(k)

    return diff_keys


# return true if differences found
def compare_attribute_group(f0, f1, name):
    got_diffs = False

    group0 = dict(f0.fid[name].attrs) if name in f0.fid else None
    group1 = dict(f1.fid[name].attrs) if name in f1.fid else None

    if (group0 is None) and (group1 is None):
        print("  %20s: no diffs (neither file has %s)" % (name, name))
    elif (group0 is None) or (group1 is None):
        # one file has group and the other doesn't
        print(
            "First file %s %s, but second file %s %s"
            % (
                "does NOT have" if group0 is None else "HAS",
                name,
                "does NOT have" if group1 is None else "HAS",
                name,
            )
        )
        got_diffs = True
    else:
        if sorted(group0.keys()) != sorted(group1.keys()):
            print("\nNames of attributes in '%s' of differ" % name)
            got_diffs = True

        # Check that the values of attributes differ
        diffs = compare_attributes(group0, group1)

        if len(diffs) > 0:
            print("\nValues of attributes in '%s' differ\n" % name)
            print("Differing attributes: ", diffs)
            got_diffs = True
        else:
            print("  %20s: no diffs" % name)


def compare_metadata(f0, f1, quiet=False, one=False, tol=1.0e-12):
    """compares metadata of two hdf files f0 and f1. Returns 0 if the files are equivalent.

    Error codes:
        10 : Times in hdf files differ
        11 : Attribute names in Info of hdf files differ
        12 : Values of attributes in Info of hdf files differ
        13 : Attribute names or values in Input of hdf files differ
        14 : Attribute names or values in Params of hdf files differ
        15 : Meta variables (Locations, VolumeLocations, LogicalLocations, Levels) differ
    """
    ERROR_TIME_DIFF = 10
    ERROR_INFO_ATTRS_DIFF = 11
    ERROR_INFO_VALUES_DIFF = 12
    ERROR_INPUT_DIFF = 13
    ERROR_PARAMS_DIFF = 14
    ERROR_META_VARS_DIFF = 15

    ret_code = 0

    # Compare the time in both files
    errTime = np.abs(f0.Time - f1.Time)
    if errTime > tol:
        print(f"Time of outputs differ by {f0.Time - f1.Time}")
        ret_code = ERROR_TIME_DIFF
        if one:
            return ret_code

    # Compare the names of attributes in /Info, except "Time"
    f0_Info = {
        key: value
        for key, value in f0.Info.items()
        if key != "Time" and key != "BlocksPerPE"
    }
    f1_Info = {
        key: value
        for key, value in f1.Info.items()
        if key != "Time" and key != "BlocksPerPE"
    }
    if sorted(f0_Info.keys()) != sorted(f1_Info.keys()):
        print("Names of attributes in '/Info' of differ")
        ret_code = ERROR_INFO_ATTRS_DIFF
        if one:
            return ret_code

    # Compare the values of attributes in /Info
    info_diffs = compare_attributes(f0_Info, f1_Info)
    if len(info_diffs) > 0:
        print("\nValues of attributes in '/Info' differ\n")
        print("Differing attributes: ", info_diffs)
        ret_code = ERROR_INFO_VALUES_DIFF
        if one:
            return ret_code
    else:
        print("  %20s: no diffs" % "Info")

    if compare_attribute_group(f0, f1, "Input"):
        ret_code = ERROR_INPUT_DIFF
        if one:
            return ret_code

    if compare_attribute_group(f0, f1, "Params"):
        ret_code = ERROR_PARAMS_DIFF
        if one:
            return ret_code

    # Now go through all variables in first file
    # and hunt for them in second file.
    #
    # Note that indices don't match when blocks
    # are different
    no_meta_variables_diff = True

    otherBlockIdx = list(f0.findBlockIdxInOther(f1, i) for i in range(f0.NumBlocks))

    for var in set(f0.Variables + f1.Variables):
        if (var not in f0.Variables) or (var not in f1.Variables):
            # we know it has to be in at least one of them
            print(
                "Variable '%s' %s the first file, but %s in the second file"
                % (
                    var,
                    "IS" if var in f0.Variables else "is NOT",
                    "IS" if var in f1.Variables else "is NOT",
                )
            )
            ret_code = ERROR_META_VARS_DIFF
            if one:
                return ret_code
            continue

        if var in ["Blocks", "Locations", "VolumeLocations"]:
            for key in f0.fid[var].keys():
                if var == "Blocks" and key == "loc.level-gid-lid-cnghost-gflag":
                    continue  # depends on number of MPI ranks and distribution of blocks among ranks

                # Compare raw data of these variables
                val0 = f0.fid[var][key]
                val1 = f1.fid[var][key]

                # Sort val1 by otherBlockIdx
                val1 = val1[otherBlockIdx]

                # Compute norm error, check against tolerance
                errVal = np.abs(val0 - val1)
                errMag = np.linalg.norm(errVal)
                if errMag > tol:
                    no_meta_variables_diff = False
                    if not quiet:
                        print("")
                    print(
                        f"Metavariable {var}/{key} differs between {f0.file} and {f1.file}"
                    )
                    if not quiet:
                        print("")
                else:
                    print("  %18s/%s: no diffs" % (var, key))
        if var in ["LogicalLocations", "Levels"]:
            # Compare raw data of these variables
            val0 = np.array(f0.fid[var])
            val1 = np.array(f1.fid[var])

            # Sort val1 by otherBlockIdx
            val1 = val1[otherBlockIdx]

            # As integers, they should be identical
            if np.any(val0 != val1):
                no_meta_variables_diff = False
                if not quiet:
                    print("")
                print(f"Metavariable {var} differs between {f0.file} and {f1.file}")
                if not quiet:
                    print("")
            else:
                print("  %20s: no diffs" % var)

    if not no_meta_variables_diff:
        ret_code = ERROR_META_VARS_DIFF
        if one:
            return ret_code

    return ret_code


def compare(
    files,
    all=False,
    brief=True,
    quiet=False,
    one=False,
    tol=1.0e-12,
    check_metadata=True,
):
    """compares two hdf files. Returns 0 if the files are equivalent.

    Error codes:
        1  : Can't open file 0
        2  : Can't open file 1
        3  : Total number of cells differ
        4  : Variable data in files differ

    Metadata Error codes:
        10 : Times in hdf files differ
        11 : Attribute names in Info of hdf files differ
        12 : Values of attributes in Info of hdf files differ
        13 : Attribute names or values in Input of hdf files differ
        14 : Attribute names or values in Params of hdf files differ
        15 : Meta variables (Locations, VolumeLocations, LogicalLocations, Levels) differ
    """

    ERROR_NO_OPEN_F0 = 1
    ERROR_NO_OPEN_F1 = 2
    ERROR_CELLS_DIFFER = 3
    ERROR_DATA_DIFFER = 4

    # **************
    # import Reader
    # **************
    from phdf import phdf

    # **************
    # Reader Help
    # **************
    # for help  on phdf uncomment following line
    # print(help(phdf))

    # Load first file and print info
    f0 = phdf(files[0])
    try:
        f0 = phdf(files[0])
        if not quiet:
            print(f0)
    except:
        print(
            """
        *** ERROR: Unable to open %s as phdf file
        """
            % files[0]
        )
        return ERROR_NO_OPEN_F0

    # Load second file and print info
    try:
        f1 = phdf(files[1])
        if not quiet:
            print(f1)
    except:
        print(
            """
        *** ERROR: Unable to open %s as phdf file
        """
            % files[1]
        )
        return ERROR_NO_OPEN_F1

    # rudimentary checks
    if f0.TotalCellsReal != f1.TotalCellsReal:
        # do both simulations have same number of cells?
        print(
            """
        These simulations have different number of cells.
        Clearly they are different.

        Quitting...
        """
        )
        return ERROR_CELLS_DIFFER

    no_diffs = True
    if check_metadata:
        if not quiet:
            print("Checking metadata")
        metadata_status = compare_metadata(f0, f1, quiet, one)
        if metadata_status != 0:
            if one:
                return metadata_status
            else:
                no_diffs = False
        else:
            if not quiet:
                print("Metadata matches")
    else:
        if not quiet:
            print("Ignoring metadata")

    # Now go through all variables in first file
    # and hunt for them in second file.
    #
    # Note that indices don't match when blocks
    # are different

    if not brief and not quiet:
        print("____Comparing on a per variable basis with tolerance %.16g" % tol)
    breakOut = False
    oneTenth = f0.TotalCells // 10
    if not quiet:
        print("Mapping indices:")
    print("Tolerance = %g" % tol)
    otherLocations = [None] * f0.TotalCells
    for idx in range(f0.TotalCells):
        if not quiet:
            if idx % oneTenth == 0:
                print("   Mapping %8d (of %d) " % (idx, f0.TotalCells))

        if f0.isGhost[idx % f0.CellsPerBlock]:
            # don't map ghost cells
            continue

        otherLocations[idx] = f0.findIndexInOther(f1, idx)
    if not quiet:
        print(f0.TotalCells, "cells mapped")

    for var in set(f0.Variables + f1.Variables):
        if var in [
            "Locations",
            "VolumeLocations",
            "LogicalLocations",
            "Levels",
            "Info",
            "Params",
            "SparseInfo",
            "Input",
            "Blocks",
        ]:
            continue

        # initialize info values
        same = True
        errMax = -1.0
        maxPos = [0, 0, 0]

        # Get values from file
        val0 = f0.Get(var)
        val1 = f1.Get(var)
        isVec = np.prod(val0.shape) != f0.TotalCells
        for idx, v in enumerate(val0):
            if f0.isGhost[idx % f0.CellsPerBlock]:
                # only consider real cells
                continue
            [ib, bidx, iz, iy, ix] = f0.ToLocation(idx)

            # find location in other file
            [idx1, ib1, bidx1, iz1, iy1, ix1] = otherLocations[idx]

            # compute error
            errVal = np.abs(v - val1[idx1])
            errMag = np.linalg.norm(errVal)

            # Note that we use norm / magnitude to compute error
            if errMag > errMax:
                errMax = errMag
                errMaxPos = [f0.x[ib, ix], f0.y[ib, iy], f0.z[ib, iz]]

            if np.linalg.norm(errVal) > tol:
                same = False
                no_diffs = False
                if brief or quiet:
                    breakOut = True
                    break

                if isVec:
                    s = "["
                    for xd in errVal:
                        if xd == 0.0:
                            s += " 0.0"
                        else:
                            s += " %10.4g" % xd
                    s += "]"
                else:
                    s = "%10.4g" % errVal
                if all:
                    print(
                        "  %20s: %6d: diff=" % (var, idx),
                        s.strip(),
                        "at:f0:%d:%.4f,%.4f,%.4f"
                        % (idx, f0.x[ib, ix], f0.y[ib, iy], f0.z[ib, iz]),
                        ":f1:%d:%.4f,%.4f,%.4f"
                        % (idx1, f1.x[ib1, ix1], f1.y[ib1, iy1], f1.z[ib1, iz1]),
                    )
        if breakOut:
            if not quiet:
                print("")
            print("Files %s and %s are different" % (f0.file, f1.file))
            if not quiet:
                print("")
            break
        if not quiet:
            if same:
                print("  %20s: no diffs" % var)
            else:
                print("____%26s: MaxDiff=%10.4g at" % (var, errMax), errMaxPos)

        if one and not same:
            break

    if no_diffs:
        return 0
    else:
        return ERROR_DATA_DIFFER


if __name__ == "__main__":
    addPath()

    # process arguments
    input = processArgs()

    brief = input.b
    quiet = input.q
    one = input.o
    ignore_metadata = input.i

    check_metadata = not ignore_metadata

    # set all only if brief not set
    if brief or quiet:
        all = False
    else:
        all = input.a
    files = input.files

    if input.tol is not None:
        tol = float(input.tol)
    else:
        tol = 1.0e-12

    if len(files) != 2:
        Usage()
        sys.exit(1)

    ret = compare(files, all, brief, quiet, one, tol, check_metadata)
    sys.exit(ret)
