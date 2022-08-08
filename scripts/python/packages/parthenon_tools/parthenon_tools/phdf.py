# =========================================================================================
# Parthenon performance portable AMR framework
# Copyright(C) 2020-2022 The Parthenon collaboration
# Licensed under the 3-clause BSD License, see LICENSE file for details
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
import h5py as h
import os, sys, errno
import numpy as np


class phdf:
    """A reader for the new HDF5 output.  Reads in a hdf5 file which
    is the only argument to the constructor.

    Class Attributes:
            Filename: Name of file that was read
                Time: Simulation time
              NCycle: Simulation cycle
             NumDims: Number of dimensions in problem run
            MaxLevel: Max level in grid
           NumBlocks: Total number of blocks
       MeshBlockSize: Size of each mesh block
       CellsPerBlock: Number of cells in each block
          TotalCells: Total number of cells in simulation
           Variables: List of variables that were written to file
                   x: Flat array of X coordinates of x cell centers
                   y: Flat array of Y coordinates of y cell centers
                   z: Flat array of Z coordinates of z cell centers

    Additional attributes for a block:
       isGhost[CellsPerBlock]: Array of size CellsPerBlock indicating ghost cells
      BlockIdx[CellsPerBlock]: Converts a cell index to [iz, iy, ix] that are
                               indices within the block for data access
       BlockBounds[NumBlocks]: Bounds of all the blocks

    Class Functions:
      Get(variable, flatten=True):
            Gets data for the named variable. Returns None if variable
            is not found in the file or the data if found.

            If variable is a vector, each element of the returned
            numpy array is a vector of that length.

            Default is to return a flat array of length TotalCells.
            However if flatten is set to False, a 4D (or 5D if vector)
            array is returned that has dimensions [NumBlocks, Nz, Ny,
            Nx] where NumBlocks is the total number of blocks, and Nz,
            Ny, and Nx are the number of cells in the z, y, and x
            directions respectively.

      ToLocation(index):
           Returns the location as an array [ib, bidx, iz, iy, ix]
           which convets the index into a block, index within that
           block, and z, y, and x locations within that block

      findIndexInOther(self,otherFile,idx,tol=1e-10):
        Given an index in my data, find the index in a different file.
        Cell centers have to match within tol.
        I've curently implemented a naive algorithm and will improve
        later

    """

    def __init__(self, filename):
        """
        Initializes a python structure with the information from
        the provided file.

        filename = name of parthenon hdf5 file
        """
        self.err = 0
        self.file = filename
        try:
            f = h.File(filename, "r")
        except:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filename)

        # Read in the timestep attributes
        info = f["Info"]
        self.fid = f
        self.NumDims = info.attrs["NumDims"]
        try:
            self.NCycle = info.attrs["NCycle"]
        except:
            self.NCycle = -1
        try:
            self.Time = info.attrs["Time"]
        except:
            self.Time = 0.0
        try:
            self.NGhost = info.attrs["NGhost"]
        except:
            self.NGhost = -1
        try:
            self.IncludesGhost = info.attrs["IncludesGhost"]
        except:
            self.IncludesGhost = 0
        self.NumBlocks = info.attrs["NumMeshBlocks"]
        self.MeshBlockSize = info.attrs["MeshBlockSize"]
        try:
            self.BlocksPerPE = info.attrs["BlocksPerPE"]
        except:
            self.BlocksPerPE = np.array((1), self.NumBlocks)
        try:
            self.OutputFormatVersion = info.attrs["OutputFormatVersion"]
        except:
            self.OutputFormatVersion = -1
        self.Coordinates = info.attrs["Coordinates"]
        self.CellsPerBlock = np.prod(self.MeshBlockSize)
        self.TotalCells = self.NumBlocks * self.CellsPerBlock

        # Read in Params (older output files don't have this, so make it optional)
        if "Params" in f:
            self.Params = dict(f["Params"].attrs)
        else:
            self.Params = None

        # Read in coordinates
        def load_coord(coord_i):
            coord_name = ["x", "y", "z"][coord_i]

            coordf = f["/Locations/" + coord_name][:, :]
            vol_loc = "/VolumeLocations/" + coord_name
            if vol_loc in f:
                coord = f[vol_loc][:, :]
            else:
                coord = np.zeros((self.NumBlocks, self.MeshBlockSize[coord_i]))
                for bId in range(self.NumBlocks):
                    for cId in range(self.MeshBlockSize[coord_i]):
                        coord[bId, cId] = 0.5 * (
                            coordf[bId, cId] + coordf[bId, cId + 1]
                        )
            return coord, coordf

        self.x, self.xf = load_coord(0)
        self.y, self.yf = load_coord(1)
        self.z, self.zf = load_coord(2)

        # fill in self.offset and block bounds
        self.offset = [0, 0, 0]
        for i in range(3):
            if self.MeshBlockSize[i] > 1:
                self.offset[i] = self.NGhost * self.IncludesGhost

        # fill in self.BlockBounds
        self.BlockBounds = [None] * self.NumBlocks
        xo = self.NGhost * self.IncludesGhost
        iOffsets = [
            self.offset[0],
            self.MeshBlockSize[0] - self.offset[0],
            self.offset[1],
            self.MeshBlockSize[1] - self.offset[1],
            self.offset[2],
            self.MeshBlockSize[2] - self.offset[2],
        ]
        eps = 1e-8
        for ib in range(self.NumBlocks):
            self.BlockBounds[ib] = [
                self.xf[ib, iOffsets[0]] - eps,
                self.xf[ib, iOffsets[1]] + eps,
                self.yf[ib, iOffsets[2]] - eps,
                self.yf[ib, iOffsets[3]] + eps,
                self.zf[ib, iOffsets[4]] - eps,
                self.zf[ib, iOffsets[5]] + eps,
            ]
        # Save info
        self.Info = dict(f["/Info"].attrs)

        # generate self.offset, isGhost and BlockIdx arrays
        self.GenAuxData()

        self.TotalCellsReal = self.NumBlocks * np.prod(
            self.MeshBlockSize - 2 * self.offset
        )

        self.MaxLevel = info.attrs["MaxLevel"]

        self.Variables = [k for k in f.keys()]
        self.varData = {k: None for k in self.Variables}

        # Construct a map of datasets to contained components,idx
        self.DatasetComponentsMap = {}
        # Construct a map of datasets to number contained components
        self.DatasetNumComponents = dict(
            zip(
                np.array(info.attrs["OutputDatasetNames"]).ravel().astype(str),
                np.array(info.attrs["NumComponents"]).ravel(),
            )
        )
        # Construct a map of components to parent datasets,idx
        self.ComponentsDatasetMap = {}
        idx_i = 0
        for dataset, num_components in self.DatasetNumComponents.items():
            num_components = num_components.astype(int)

            component_names = (
                np.array(info.attrs["ComponentNames"])
                .ravel()
                .astype(str)[idx_i : idx_i + num_components]
            )
            component_indices = list(range(num_components))

            self.DatasetComponentsMap[dataset] = dict(
                zip(component_names, component_indices)
            )

            # Note: if a component is in multiple datasets, then this map
            # will point to the last dataset that contains it
            self.ComponentsDatasetMap.update(
                {
                    name: (dataset, idx)
                    for name, idx in zip(component_names, component_indices)
                }
            )

            idx_i += num_components

    def GenAuxData(self):
        """
        Additional attributes filled in by function GenAuxData():
                          offset[3]: Offsets for real cells in the block
             isGhost[CellsPerBlock]: Array of size CellsPerBlock indicating ghost cells
            BlockIdx[CellsPerBlock]: Converts a cell index to [ib, bidx, iz, iy, ix]
                                     into [ a block ID, index within that
                                     block, and z, y, and x locations within that block]
             BlockBounds[NumBlocks]: Bounds of all the blocks
        """
        # flag for ghost cells.
        # Logic is easier starting with all ghost and unmarking
        self.offset = np.zeros(3, "i")
        for i in range(3):
            if self.MeshBlockSize[i] > 1:
                self.offset[i] = self.NGhost * self.IncludesGhost
        xRange = range(self.MeshBlockSize[0])
        yRange = range(self.MeshBlockSize[1])
        zRange = range(self.MeshBlockSize[2])
        xo = [self.offset[0], self.MeshBlockSize[0] - self.offset[0]]
        yo = [self.offset[1], self.MeshBlockSize[1] - self.offset[1]]
        zo = [self.offset[2], self.MeshBlockSize[2] - self.offset[2]]

        self.BlockIdx = [None] * self.CellsPerBlock
        self.isGhost = np.ones(self.CellsPerBlock, dtype=bool)
        index = 0
        yMask = False
        zMask = False
        for k in zRange:
            if self.NumDims > 2:
                zMask = k < zo[0] or k >= zo[1]
            for j in yRange:
                if self.NumDims > 1:
                    yMask = j < yo[0] or j >= yo[1]
                for i in xRange:
                    xMask = i < xo[0] or i >= xo[1]
                    self.isGhost[index] = xMask or yMask or zMask
                    self.BlockIdx[index] = [k, j, i]
                    index += 1

    def ToLocation(self, index):
        """
        Converts an index to [blockID, bidx, Z, Y, X]
        """

        ib = int(index / self.CellsPerBlock)
        bidx = index % self.CellsPerBlock
        [iz, iy, ix] = self.BlockIdx[bidx]
        return [ib, bidx, iz, iy, ix]

    #    def isPointInBlock(self,
    def findIndexInOther(self, other, idx, tol=1e-10, verbose=0):
        """
        Given an index in my data, find the index in a different file.
        Cell centers have to match within tol.
        I've curently implemented a naive algorithm and will improve
        later
        """
        [ib, bidx, iz, iy, ix] = self.ToLocation(idx)

        (myX, myY, myZ) = (self.x[ib, ix], self.y[ib, iy], self.z[ib, iz])

        # now hunt in other file
        for ib1 in range(other.NumBlocks):
            ib1Bounds = other.BlockBounds[ib1]
            # compute deltas
            # see if block bounds match
            if myX < ib1Bounds[0] or myX > ib1Bounds[1]:
                continue
            if self.NumDims > 1 and (myY < ib1Bounds[2] or myY > ib1Bounds[3]):
                continue
            if self.NumDims > 2 and (myZ < ib1Bounds[4] or myZ > ib1Bounds[5]):
                continue

            # smart finder:
            deltas = [other.x[ib1][1] - other.x[ib1][0], 1.0, 1.0]
            if other.MeshBlockSize[1] > 1:
                deltas[1] = other.y[ib1][1] - other.y[ib1][0]
            if other.MeshBlockSize[2] > 1:
                deltas[2] = other.z[ib1][1] - other.z[ib1][0]

            ix1 = (
                int(round((myX - other.x[ib1][other.offset[0]]) / deltas[0]))
                + other.offset[0]
            )
            if self.NumDims > 1:
                iy1 = (
                    int(round((myY - other.y[ib1][other.offset[1]]) / deltas[1]))
                    + other.offset[1]
                )
            else:
                iy1 = 0

            if self.NumDims > 2:
                iz1 = (
                    int(round((myZ - other.z[ib1][other.offset[2]]) / deltas[2]))
                    + other.offset[2]
                )
            else:
                iz1 = 0

            otherX, otherY, otherZ = (
                other.x[ib1, ix1],
                other.y[ib1, iy1],
                other.z[ib1, iz1],
            )
            if (
                abs(myX - otherX) > tol
                or abs(myY - otherY) > tol
                or abs(myZ - otherZ) > tol
            ):
                print("skipping:", ib1, [ix1, iy1, iz1], [otherX, otherY, otherX])
                continue

            iCell1 = ix1 + other.MeshBlockSize[0] * (iy1 + iz1 * other.MeshBlockSize[1])
            idx1 = ib1 * other.CellsPerBlock + iCell1
            return [idx1, ib1, iCell1, iz1, iy1, ix1]

        if verbose:
            print("ox=")
            for xx in other.x:
                print("    ", ["%.20lf" % x for x in xx])

            print("oy=")
            for yy in other.y:
                print("    ", ["%.20lf" % y for y in yy])

            print("oz=")
            for zz in other.z:
                print("    ", ["%.20lf" % z for z in zz])

            print("deltas=", deltas)
            print("bounds:")
            for b in other.BlockBounds:
                print(b)
                print("me=", idx, self.isGhost[bidx], [ib, bidx, iz, iy, ix])
                print("LOOKING:", idx, ["%.20lf" % x for x in [myX, myY, myZ]])

        raise ValueError("Unable to map cell")

    def findBlockIdxInOther(self, other, ib, tol=1e-10, verbose=False):
        """
        Given an meshblock index in my data, find the meshblock index in a different file.
        """

        myibBounds = np.array(self.BlockBounds[ib])

        # now hunt in other file
        for ib1 in range(other.NumBlocks):
            ib1Bounds = np.array(other.BlockBounds[ib1])

            if np.all(np.abs(myibBounds - ib1Bounds) < tol):
                return ib1

        if verbose:
            print(f"Block id: {ib} with bounds {myibBounds} not found in {other.file}")
        return None  # block index not found

    def Get(self, variable, flatten=True):
        """
        Reads data for the named variable from file.

        Returns None if variable is not found in the file or the data
        if found.

        If variable is a vector, each element of the returned numpy
        array is a vector of that length.

        Default is to return a flat array of length TotalCells.
        However if flatten is set to False, a 4D (or 5D if vector)
        array is returned that has dimensions [NumBlocks, Nz, Ny, Nx]
        where NumBlocks is the total number of blocks, and Nz, Ny, and
        Nx are the number of cells in the z, y, and x directions
        respectively.

        """
        try:
            if self.varData[variable] is None:
                self.varData[variable] = self.fid[variable][:]
                vShape = self.varData[variable].shape
                if self.OutputFormatVersion == -1:
                    vLen = vShape[-1]
                else:
                    vLen = vShape[1]  # index 0 is the block, so we need to use 1
                # if variable is a scalar remove the component index
                if vLen == 1:
                    tmp = self.varData[variable].reshape(self.TotalCells)
                    newShape = (
                        self.NumBlocks,
                        self.MeshBlockSize[2],
                        self.MeshBlockSize[1],
                        self.MeshBlockSize[0],
                    )
                    self.varData[variable] = tmp.reshape((newShape))

        except:
            print(
                """
            ERROR: Unable to read %s from file %s
            """
                % (variable, self.file)
            )
            return None

        vShape = self.varData[variable].shape
        if flatten:
            # if variable is not a scalar flatten cells component-wise
            if np.prod(vShape) > self.TotalCells:
                if self.OutputFormatVersion == -1:
                    return self.varData[variable][:].reshape(
                        self.TotalCells, vShape[-1]
                    )

                ret = np.empty(
                    (vShape[1], self.TotalCells), dtype=self.varData[variable].dtype
                )
                ret[:] = np.nan
                for i in range(len(vShape[1])):
                    ret[i, :] = self.varData[variable][:, i, :, :, :]
                assert (ret != np.nan).all()
                return ret
            else:
                return self.varData[variable][:].reshape(self.TotalCells)

        return self.varData[variable][:]

    def GetComponents(self, components, flatten=True):
        """
        Reads data for the named components from file.

        Returns components as a dictionary of {component:component_data}

        Throws an exception if the components do not exist in the file

        Default is to return a flat array of length TotalCells.  However if
        flatten is set to False, a 4D for each component array is returned that
        has dimensions [NumBlocks, Nz, Ny, Nx] where NumBlocks is the total
        number of blocks, and Nz, Ny, and Nx are the number of cells in the z,
        y, and x directions respectively.

        Components can be from a single variable or spread across multiple variables.

        Components refer to top-level components of variables. The number of
        components and the names of each component in variables are specified in
        the phdf file in Info/NumComponents and Info/ComponentNames with
        variables in the order of Info/OutputDatasetNames.

        In the c++ code, components of variables can be named by supplying a
        std::vector of std::strings when creating the Metadata for variables.
        Without specifying component names, by default variables have a single
        component with the same name as the variable. See the Parthenon docs
        outputs.md#preparing-outputs-for-yt for details.
        """

        # Check if these components exist
        missing_components = [
            component
            for component in components
            if not component in self.ComponentsDatasetMap
        ]
        if len(missing_components) > 0:
            raise Exception(
                f"Components {missing_components} are missing from {self.file}"
            )

        # Determine which Datasets contain components
        dataset_names = set(
            self.ComponentsDatasetMap[component][0] for component in components
        )

        # Start a map of required components
        component_data = {}

        for dataset_name in dataset_names:
            dataset = self.Get(dataset_name, flatten=flatten)

            for component, idx in self.DatasetComponentsMap[dataset_name].items():
                if component in components:
                    if self.DatasetNumComponents[dataset_name] == 1:
                        # If dataset isn't a vector, just save dataset
                        component_data[component] = dataset
                    else:
                        # Data is a vector, save only the component
                        if self.OutputFormatVersion == -1:
                            component_data[component] = dataset[..., idx]
                        else:
                            if flatten:
                                component_data[component] = dataset[idx, ...]
                            # need to take leading block index into account
                            else:
                                component_data[component] = dataset[:, idx, ...]

        return component_data

    def __GenLocMeshGrid(self, z, y, x, flatten):
        """
        Returns Z[grid_idx,k,j,i], Y[grid_idx,k,j,i], X[grid_idx,k,j,i] full
        arrays of locations from 2D arrays z[grid_idx,k], y[grid_idx,j],
        x[grid_idx,i]
        """

        if not (x.shape[0] == y.shape[0] == z.shape[0]):
            raise Exception("z,y,x have different number of grids")

        # loc[grid_idx,k,j,i]
        loc_shape = (x.shape[0], z.shape[1], y.shape[1], x.shape[1])

        Z = np.empty(loc_shape)
        Y = np.empty(loc_shape)
        X = np.empty(loc_shape)

        for grid_idx in range(loc_shape[0]):
            Z[grid_idx], Y[grid_idx], X[grid_idx] = np.meshgrid(
                z[grid_idx], y[grid_idx], x[grid_idx], indexing="ij"
            )

        if flatten:
            Z = Z.ravel()
            Y = Y.ravel()
            X = X.ravel()

        return Z, Y, X

    def GetVolumeLocations(self, flatten=True):
        """
        Returns Z,Y,X arrays of volume centered locations in the dataset

        Default is to return a flat array of length TotalCells.  However if
        flatten is set to False, a 4D for each dimension is returned that has
        dimensions [NumBlocks, Nz, Ny, Nx] where NumBlocks is the total number
        of blocks, and Nz, Ny, and Nx are the number of cells in the z, y, and
        x directions respectively.
        """

        return self.__GenLocMeshGrid(self.z, self.y, self.x, flatten)

    def GetFaceLocations(self, flatten=True):
        """
        Returns Z,Y,X arrays of face centered locations in the dataset

        Default is to return a flat array of length TotalCells.  However if
        flatten is set to False, a 4D for each dimension is returned that has
        dimensions [NumBlocks, Nz, Ny, Nx] where NumBlocks is the total number
        of blocks, and Nz, Ny, and Nx are the number of cells in the z, y, and
        x directions respectively.
        """

        return self.__GenLocMeshGrid(self.zf, self.yf, self.xf, flatten)

    def __str__(self):
        return (
            """
-------------------------------------------
    Filename=%s
-------------------------------------------
                Time=%.4lf
              NCycle=%d
             NumDims=%d
            MaxLevel=%d
           NumBlocks=%d
       MeshBlockSize=%s
       CellsPerBlock=%d
          TotalCells=%d
      TotalCellsReal=%d
               NumPE=%d
         BlocksPerPE=%s
              NGhost=%d
       IncludesGhost=%d
         Coordinates=%s
 OutputFormatVersion=%d
--------------------------------------------
           Variables="""
            % (
                self.file,
                self.Time,
                self.NCycle,
                self.NumDims,
                self.MaxLevel,
                self.NumBlocks,
                self.MeshBlockSize,
                self.CellsPerBlock,
                self.TotalCells,
                self.TotalCellsReal,
                np.sum(self.BlocksPerPE.shape),
                self.BlocksPerPE,
                self.NGhost,
                self.IncludesGhost,
                self.Coordinates,
                self.OutputFormatVersion,
            )
            + str([k for k in self.Variables])
            + """
--------------------------------------------
"""
        )


if __name__ == "__main__":
    files = sys.argv[1:]
    for filename in files:
        ba = phdf(filename)
        print(ba)
        l = ba.Get("c.c.bulk.momentum")
        print("cmom=", l.shape)
        l = ba.Get("c.c.bulk.bulk_modulus")
        print("mod=", l.shape)
        print(help(ba))
