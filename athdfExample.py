#=========================================================================================
# (C) (or copyright) 2020. Triad National Security, LLC. All rights reserved.
#
# This program was produced under U.S. Government contract 89233218CNA000001 for Los
# Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
# for the U.S. Department of Energy/National Nuclear Security Administration. All rights
# in the program are reserved by Triad National Security, LLC, and the U.S. Department
# of Energy/National Nuclear Security Administration. The Government is granted for
# itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
# license in this material to reproduce, prepare derivative works, distribute copies to
# the public, perform publicly and display publicly, and to permit others to do so.
#=========================================================================================

from __future__ import print_function
#****************************************************************
# Note: reader import occurs after we fix the path at the bottom
#****************************************************************

#**************
# other imports
import os
import sys
import numpy as np
import argparse

def Usage():
    print("""

    Usage: %s [-quiet] [-brief] [-all] [-one] [--tol=eps] file1.hdf5 file2.hdf5

           -all: report all diffs at all positions
           -one: Quit after first different variable
         -brief: Only report if files are different
                 Overrides --all
         -quiet: Only report if files are different and
                 don't print any extraneous info.
      --tol=eps: set tolerance to eps.  Default 1.0e-12

    This example takes two hdf files and compares them to see if there are
    differences in the state variables.

    Shows how to load up a structure and prints one of the structures

"""%os.path.basename(__file__)
    )

def processArgs():
    parser = argparse.ArgumentParser(description="""
    arguments for differencing script
    """)
    parser.add_argument('-a', '-all', action='store_true', help='report all diffs at all positions')
    parser.add_argument('-t', '--tol', action='store', help='Sets tolerance for comparisons.  Default 1e-12')
    parser.add_argument('-o', '-one', action='store_true', help='Only report data for first different variable.')
    parser.add_argument('-b', '-brief', action='store_true', help='Only report if files are different.  Overrides -all')
    parser.add_argument('-q', '-quiet', action='store_true', help='Only report if files are different.  No other output. Overrides -all')
    parser.add_argument('files', nargs='*')

    return parser.parse_args()


def addPath():
    """ add the vis/python directory to the pythonpath variable """
    myPath = os.path.realpath(os.path.dirname(__file__))
    sys.path.insert(0,myPath+'/../vis/python')
    sys.path.insert(0,myPath+'/vis/python')

if __name__ == "__main__":
    addPath()

    #**************
    # import Reader
    #**************
    from better_athdf import better_athdf as bhdf


    #**************
    # Reader Help
    #**************
    # for help  on better_athdf uncomment following line
    # print(help(bhdf))

    # process arguments
    input = processArgs()

    brief=input.b
    quiet=input.q
    one = input.o

    # set all only if brief not set
    if brief or quiet:
        all=False
    else:
        all = input.a
    files = input.files

    if input.tol is not None:
        tol = float(input.tol)
    else:
        tol = 1.0e-12


    if len(files) != 2:
        Usage()
        exit(1)

    # Load first file and print info
    try:
        f0 = bhdf(files[0])
        if not quiet: print(f0)
    except:
        exit(1)

    # Load second file and print info
    try:
        f1 = bhdf(files[1])
        if not quiet:  print(f1)
    except:
        print("""
        *** ERROR: Unable to open %s as athdf file
        """%files[1])
        exit(2)

    # rudimentary checks
    if f0.TotalCellsReal != f1.TotalCellsReal:
        # do both simulations have same number of cells?
        print("""
        These simulations have different number of cells.
        Clearly they are different.

        Quitting...
        """)
        exit(3)

    # Now go through all variables in first file
    # and hunt for them in second file.
    #
    # Note that indices don't match when blocks
    # are different
    no_diffs = True

    if not brief and not quiet:
        print('____Comparing on a per variable basis with tolerance %.16g'%tol)
    breakOut = False
    oneTenth = f0.TotalCells/10
    if not quiet: print('Mapping indices:')
    print('Tolerance = %g' % tol)
    otherLocations = [None]*f0.TotalCells
    for idx in range(f0.TotalCells):
        if not quiet:
            if idx%oneTenth == 0:
                print('   Mapping %8d (of %d) '%(idx,f0.TotalCells))

        if f0.isGhost[idx%f0.CellsPerBlock]:
            # don't map ghost cells
            continue

        otherLocations[idx] = f0.findIndexInOther(f1,idx)
    if not quiet: print(f0.TotalCells,'cells mapped')

    for var in f0.Variables:
        if var == 'Locations' or var == 'Timestep':
            continue
        #initialize info values
        same = True
        errMax = -1.0
        maxPos=[0,0,0]

        # Get values from file
        val0 = f0.Get(var)
        val1 = f1.Get(var)
        isVec = np.prod(val0.shape) != f0.TotalCells
        for idx,v in enumerate(val0):
            if f0.isGhost[idx%f0.CellsPerBlock]:
                # only consider real cells
                continue
            [ib,bidx,iz,iy,ix] = f0.ToLocation(idx)

            # find location in other file
            [idx1, ib1, bidx1, iz1, iy1, ix1] = otherLocations[idx]

            # compute error
            errVal = np.abs(v - val1[idx1])
            errMag = np.linalg.norm(errVal)

            # Note that we use norm / magnitude to compute error
            if errMag > errMax:
                errMax = errMag
                errMaxPos = [f0.x[ib,ix], f0.y[ib,iy], f0.z[ib,iz]]

            if np.linalg.norm(errVal) > tol:
                same = False
                no_diffs = False
                if brief or quiet:
                    breakOut=True
                    break

                if isVec:
                    s='['
                    for xd in errVal:
                        if xd == 0.:
                            s += ' 0.0'
                        else:
                            s += ' %10.4g'%xd
                    s+= ']'
                else:
                    s = '%10.4g'%errVal
                if all:
                    print('  %20s: %6d: diff='%(var,idx),s.strip(),
                          'at:f0:%d:%.4f,%.4f,%.4f'%(idx,
                                                     f0.x[ib,ix],
                                                     f0.y[ib,iy],
                                                     f0.z[ib,iz]),
                          ':f1:%d:%.4f,%.4f,%.4f'%(idx1,f1.x[ib1,ix1],f1.y[ib1,iy1],f1.z[ib1,iz1]))
        if breakOut:
            if not quiet: print("")
            print('Files %s and %s are different'%(f0.file, f1.file))
            if not quiet: print("")
            break
        if not quiet:
            if same:
                print('  %20s: no diffs'%var)
            else:
                print('____%26s: MaxDiff=%10.4g at'%(var,errMax),errMaxPos)

        if one and not same:
            break

    if no_diffs:
      exit(0)
    else:
      exit(4)
