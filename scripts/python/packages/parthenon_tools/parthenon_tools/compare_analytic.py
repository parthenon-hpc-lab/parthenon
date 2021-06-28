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

import numpy as np


def norm_err_func(gold, test, norm_ord=2, relative=False, ignore_gold_zero=True):
    """Example error metric function
    Arguments:
    gold: array_like
        Array of gold data for a single component (Supplied by analytic_component)
    test: 1D array
        1D array of test data

    Keyword Arguments:
    norm_ord=2: {non-zero int, inf, -inf, ‘fro’, ‘nuc’}
        Order of the norm. See numpy.linalg.norm for documentation of norms
    relative=False: boolean
        If true, at every point divides the difference between the gold and test
        data by the average of the gold and test data
    ignore_gold_zero=True: boolean
        If true and relative is also true, then ignores differences between gold
        and test where the gold data is zero. Note that If ignore_gold_zero is
        false and relative is true, then every difference between test and gold
        where gold is zero will have a weight of 2

    Returns error metric between gold and test
    """
    err_val = np.abs(gold - test)

    if relative:
        denom = 0.5 * (np.abs(gold) + np.abs(test))
        # When test==0 but gold!=0 or vice versa, use the mean of the entire
        # data set as denom to avoid giving these points an err_val=2.0
        denom[np.logical_or(gold == 0, test == 0)] = 0.5 * np.mean(
            np.abs(gold) + np.abs(test)
        )
        # To avoid nan when gold and test are 0
        denom[denom == 0] = 1

        err_val /= denom

        if ignore_gold_zero:
            err_val = err_val[gold != 0]

    return np.linalg.norm(err_val, ord=norm_ord)


################################################################################
#   Compare_analytic
#
#   Compares data in filename to a dictionary of analytic functions for some
#   components
################################################################################


def compare_analytic(
    filename, analytic_components, err_func=norm_err_func, tol=1e-12, quiet=False
):
    """Compares data in filename to analytic gold data in analytic_components.

    Arguments:
    filename: string
        Filename of ".phdf" file to compare data
    analytic_components: dictionary
        Dictionary keying component names to analytic functions.
        Each analytic function in the dictionary takes the arguments:

        def analytic_func(Z,Y,X,t)

        where Z,Y,X comprise arrays of z,y,x coords to compute the analytic solution
        for a component at time t

    Keyword Arguments:
    err_func=norm_err_func: function(gold,test)
        Error function that accepts the analytic solution and
        data and returns an error metric. The default is the L2 norm of the
        difference between the gold and test data. Other order norms and
        relative error functions can be constructed from norm_err_func or can be
        created from scratch
    tol=1e-12: float
        Tolerance of error metric.
    quiet=False: boolean
        Set to true to supress printing errors exceeding tol

    Returns True if the error of all components is under the tolerance, otherwise
    returns False.
    """

    try:
        import phdf
    except ModuleNotFoundError:
        print("Couldn't find module to read Parthenon hdf5 files.")
        return False

    datafile = phdf.phdf(filename)

    # Dictionary of component_name:component[grid_idx,k,j,i]
    file_components = datafile.GetComponents(analytic_components.keys(), flatten=False)

    # Generate location arrays for each grid
    Z, Y, X = datafile.GetVolumeLocations()

    # Check all components for which an analytic version exists
    all_ok = True
    for component in analytic_components.keys():

        # Compute the analytic component at Z,Y,X
        analytic_component = analytic_components[component](Z, Y, X, datafile.Time)

        # Compute the error between the file and analytic component
        err = err_func(analytic_component, file_components[component].ravel())

        if err > tol:
            if not quiet:
                print(
                    f"Component {component} in {filename} error {err} exceeds tolerance {tol}"
                )
            all_ok = False

    return all_ok
