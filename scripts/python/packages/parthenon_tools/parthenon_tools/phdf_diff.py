# =========================================================================================
# (C) (or copyright) 2020-2022. Triad National Security, LLC. All rights reserved.
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

    Usage: %s [-quiet] [-brief] [-one] [--tol=eps] [-ignore_metadata] [-check_input] file1.phdf file2.phdf

                  -one: Quit after first different variable
                -brief: Only report if files are different
                -quiet: Only report if files are different and
                        don't print any extraneous info.
             --tol=eps: set tolerance to eps.  Default 1.0e-12
      -ignore_metadata: Ignore differences in metadata
          -check_input: Include the Input metadata in comparison (default is off)
             -relative: Compare relative differences using the
                        first file as the reference. Ignores
                        points where the first file is zero

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
        "-t",
        "--tol",
        action="store",
        help="Sets tolerance for comparisons. Default 1e-12",
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
        help="Only report if files are different.",
    )
    parser.add_argument(
        "-q",
        "-quiet",
        action="store_true",
        help="Only report if files are different. No other output.",
    )
    parser.add_argument(
        "-i",
        "-ignore_metadata",
        action="store_true",
        help="Ignore differences in metadata.",
    )
    parser.add_argument(
        "-check_input",
        action="store_true",
        help="Include the Input metadata in comparison.",
    )
    parser.add_argument(
        "-r", "-relative", action="store_true", help="Compare relative differences."
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
    PARAM_TOL = 1e-12
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
            if a.dtype == b.dtype == np.float64:
                if np.any(np.abs(a - b) >= PARAM_TOL):
                    diff_keys.append(k)
            elif np.any(a != b):
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

            print("\nFirst file:")
            for k in diffs:
                print("%20s: " % k, group0[k] if k in group0 else "<does not exist>")

            print("\nSecond file:")
            for k in diffs:
                print("%20s: " % k, group1[k] if k in group1 else "<does not exist>")
        else:
            print("  %20s: no diffs" % name)

    return got_diffs


def compare_metadata(f0, f1, quiet=False, one=False, check_input=False, tol=1.0e-12):
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
        if key != "Time"
        and key != "BlocksPerPE"
        and key != "WallTime"
        and key != "OutputFormatVersion"
    }
    f1_Info = {
        key: value
        for key, value in f1.Info.items()
        if key != "Time"
        and key != "BlocksPerPE"
        and key != "WallTime"
        and key != "OutputFormatVersion"
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

    if check_input:
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
                err_val = np.abs(val0 - val1)
                err_mag = np.linalg.norm(err_val)
                if err_mag > tol:
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
        if var in ["LogicalLocations", "Levels", "SparseInfo"]:
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
    brief=False,
    quiet=False,
    one=False,
    tol=1.0e-12,
    check_metadata=True,
    check_input=False,
    relative=False,
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
    except Exception as e:
        print(
            """
        *** ERROR: Unable to open %s as phdf file
        """
            % files[0]
        )
        print(repr(e))
        return ERROR_NO_OPEN_F0

    # Load second file and print info
    try:
        f1 = phdf(files[1])
        if not quiet:
            print(f1)
    except Exception as e:
        print(
            """
        *** ERROR: Unable to open %s as phdf file
        """
            % files[1]
        )
        print(repr(e))
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
        metadata_status = compare_metadata(f0, f1, quiet, one, check_input)
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
    print("Tolerance = %g" % tol)

    # Make loc array of locations matching the shape of val0,val1
    loc = f0.GetVolumeLocations(flatten=False)

    for var in set(f0.Variables + f1.Variables):
        var_no_diffs = True
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

        # Get values from file
        val0 = f0.Get(var, flatten=False)
        val1 = f1.Get(var, flatten=False)

        is_vec = np.prod(val0.shape) != f0.TotalCells

        # Determine arrangement of mesh blocks of f1 in terms of ordering in f0
        otherBlockIdx = list(f0.findBlockIdxInOther(f1, i) for i in range(f0.NumBlocks))

        # Rearrange val1 to match ordering of meshblocks in val0
        val1 = val1[otherBlockIdx]

        # compute error at every point
        if relative:
            denom = 0.5 * (np.abs(val0) + np.abs(val1))
            # When val0==0 but val1!=0 or vice versa, use the mean of the
            # entire data set as denom to avoid giving these points an
            # err_val=2.0
            denom[np.logical_or(val0 == 0, val1 == 0)] = 0.5 * np.mean(
                np.abs(val0) + np.abs(val1)
            )

            err_val = np.abs(val0 - val1) / denom
            # Set error values where denom==0 to 0
            # Numpy masked arrays would be more robust here, but they are very slow
            err_val[denom == 0] = 0
        else:
            err_val = np.abs(val0 - val1)

        # Compute magnitude of error at every point
        if is_vec:
            # Norm every vector
            err_mag = np.linalg.norm(err_val, axis=-1)
        else:
            # Just plain error for scalars
            err_mag = err_val
        err_max = err_mag.max()

        # Check if the error of any block exceeds the tolerance
        if err_max > tol:
            no_diffs = False
            var_no_diffs = False

            if quiet:
                continue  # Skip reporting the error

            if one:
                # Print the maximum difference only
                bad_idx = np.argmax(err_mag)
                bad_idx = np.array(np.unravel_index(bad_idx, err_mag.shape))

                # Reshape for printing step
                bad_idxs = bad_idx.reshape((1, *bad_idx.shape))
            else:
                # Print all differences exceeding maximum
                bad_idxs = np.argwhere(err_mag > tol)

            for bad_idx in bad_idxs:
                bad_idx = tuple(bad_idx)

                # Find the bad location
                bad_loc = np.array(loc)[
                    :, bad_idx[0], bad_idx[1], bad_idx[2], bad_idx[3]
                ]

                # TODO(forrestglines): Check that the bkji and zyx reported are the correct order
                print(f"Diff in {var:20s}")
                print(
                    f"    bkji: ({bad_idx[0]:4d},{bad_idx[1]:4d},{bad_idx[2]:4d},{bad_idx[3]:4d})"
                )
                print(f"    zyx: ({bad_loc[0]:4f},{bad_loc[1]:4f},{bad_loc[2]:4f})")
                print(f"    err_mag: {err_mag[bad_idx]:4f}")
                if is_vec:
                    print(f"    f0: " + " ".join(f"{u:.4e}" for u in val0[bad_idx]))
                    print(f"    f1: " + " ".join(f"{u:.4e}" for u in val1[bad_idx]))
                    print(f"    err: " + " ".join(f"{u:.4e}" for u in err_val[bad_idx]))
                else:
                    print(f"    f0: {val0[bad_idx]:.4e}")
                    print(f"    f1: {val1[bad_idx]:.4e}")
        if not quiet:
            if var_no_diffs:
                print(f"  {var:20s}: no diffs")
            else:
                print(f"  {var:20s}: differs")
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
    check_input = input.check_input
    relative = input.r
    check_metadata = not ignore_metadata
    files = input.files

    if input.tol is not None:
        tol = float(input.tol)
    else:
        tol = 1.0e-12

    if len(files) != 2:
        Usage()
        sys.exit(1)

    ret = compare(files,
        brief,
        quiet,
        one,
        tol,
        check_metadata,
        check_input,
        relative,
    )
    sys.exit(ret)
