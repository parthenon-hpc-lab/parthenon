# ========================================================================================
# Parthenon performance portable AMR framework
# Copyright(C) 2020-2026 The Parthenon collaboration
# Licensed under the 3-clause BSD License, see LICENSE file for details
# ========================================================================================
# (C) (or copyright) 2020-2026. Triad National Security, LLC. All rights reserved.
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

# This file was modified with the assistance of generative AI.

# Modules
import os
import sys
import xml.etree.ElementTree as ET

import h5py
import utils.test_case

# To prevent littering up imported folders with .pyc files or __pycache_ folder
sys.dont_write_bytecode = True


def check_xdmf_geometry(hdf_filename, expected_num_dims):
    """Check that an XDMF mesh description agrees with its HDF5 file."""
    xdmf_filename = hdf_filename + ".xdmf"

    try:
        tree = ET.parse(xdmf_filename)
        with h5py.File(hdf_filename, "r") as hdf_file:
            info = hdf_file["Info"]
            num_dims = int(info.attrs["NumDims"])
            num_blocks = int(info.attrs["NumMeshBlocks"])
            mesh_block_size = tuple(int(n) for n in info.attrs["MeshBlockSize"])
            location_shapes = {
                axis: hdf_file[f"Locations/{axis}"].shape
                for axis in ("x", "y", "z")[:num_dims]
            }
    except (ET.ParseError, KeyError, OSError) as error:
        print(f"ERROR: Could not read XDMF/HDF5 geometry: {error}")
        return False

    success = True
    if num_dims != expected_num_dims:
        print(
            f"ERROR: Expected a {expected_num_dims}D HDF5 mesh, but {hdf_filename} "
            f"reports NumDims={num_dims}."
        )
        success = False

    if num_dims not in (2, 3):
        print(f"ERROR: XDMF geometry check does not support NumDims={num_dims}.")
        return False

    collection = tree.find("./Domain/Grid")
    if collection is None:
        print(f"ERROR: Could not find the mesh collection in {xdmf_filename}.")
        return False

    grids = collection.findall("./Grid[@GridType='Uniform']")
    if len(grids) != num_blocks:
        print(
            f"ERROR: {xdmf_filename} contains {len(grids)} mesh blocks, but "
            f"{hdf_filename} reports {num_blocks}."
        )
        success = False

    expected_topology_type = f"{num_dims}DRectMesh"
    expected_topology_dims = tuple(n + 1 for n in reversed(mesh_block_size[:num_dims]))
    expected_geometry_type = {2: "VXVY", 3: "VXVYVZ"}[num_dims]
    axes = ("x", "y", "z")[:num_dims]

    for block_index, grid in enumerate(grids):
        topology = grid.find("Topology")
        geometry = grid.find("Geometry")
        block_name = grid.get("Name", str(block_index))

        if topology is None or geometry is None:
            print(f"ERROR: XDMF block {block_name} is missing topology or geometry.")
            success = False
            continue

        topology_type = topology.get("TopologyType")
        if topology_type != expected_topology_type:
            print(
                f"ERROR: XDMF block {block_name} has TopologyType={topology_type}; "
                f"expected {expected_topology_type}."
            )
            success = False

        try:
            topology_dims = tuple(
                int(n) for n in topology.get("Dimensions", "").split()
            )
        except ValueError:
            topology_dims = ()
        if topology_dims != expected_topology_dims:
            print(
                f"ERROR: XDMF block {block_name} has topology dimensions "
                f"{topology_dims}; expected {expected_topology_dims} from the HDF5 "
                f"MeshBlockSize {mesh_block_size}."
            )
            success = False

        geometry_type = geometry.get("GeometryType")
        if geometry_type != expected_geometry_type:
            print(
                f"ERROR: XDMF block {block_name} has GeometryType={geometry_type}; "
                f"expected {expected_geometry_type}."
            )
            success = False

        coordinate_slabs = geometry.findall("./DataItem")
        if len(coordinate_slabs) != num_dims:
            print(
                f"ERROR: XDMF block {block_name} contains {len(coordinate_slabs)} "
                f"coordinate arrays; expected {num_dims}."
            )
            success = False

        for axis_index, (axis, slab) in enumerate(zip(axes, coordinate_slabs)):
            expected_slab_dims = (mesh_block_size[axis_index] + 1,)
            try:
                slab_dims = tuple(int(n) for n in slab.get("Dimensions", "").split())
            except ValueError:
                slab_dims = ()
            if slab_dims != expected_slab_dims:
                print(
                    f"ERROR: XDMF block {block_name} coordinate {axis} has dimensions "
                    f"{slab_dims}; expected {expected_slab_dims}."
                )
                success = False

            selector_items = [
                item for item in slab.findall("DataItem") if item.get("Format") == "XML"
            ]
            expected_selector = (
                block_index,
                0,
                1,
                1,
                1,
                expected_slab_dims[0],
            )
            try:
                selector = tuple(int(n) for n in selector_items[0].text.split())
            except (AttributeError, IndexError, ValueError):
                selector = ()
            if len(selector_items) != 1 or selector != expected_selector:
                print(
                    f"ERROR: XDMF block {block_name} coordinate {axis} has hyperslab "
                    f"selector {selector}; expected {expected_selector}."
                )
                success = False

            hdf_items = [
                item for item in slab.findall("DataItem") if item.get("Format") == "HDF"
            ]
            if len(hdf_items) != 1:
                print(
                    f"ERROR: XDMF block {block_name} coordinate {axis} does not have "
                    f"exactly one HDF5 reference."
                )
                success = False
                continue

            hdf_item = hdf_items[0]
            try:
                hdf_dims = tuple(int(n) for n in hdf_item.get("Dimensions", "").split())
            except ValueError:
                hdf_dims = ()
            if hdf_dims != location_shapes[axis]:
                print(
                    f"ERROR: XDMF block {block_name} coordinate {axis} reports HDF5 "
                    f"dimensions {hdf_dims}; dataset Locations/{axis} has shape "
                    f"{location_shapes[axis]}."
                )
                success = False

            expected_reference = f"{os.path.basename(hdf_filename)}:/Locations/{axis}"
            if (hdf_item.text or "").strip() != expected_reference:
                print(
                    f"ERROR: XDMF block {block_name} coordinate {axis} references "
                    f"{(hdf_item.text or '').strip()}; expected {expected_reference}."
                )
                success = False

    return success


class TestCase(utils.test_case.TestCaseAbs):
    def Prepare(self, parameters, step):
        # enable coverage testing on pass where restart
        # files are both read and written
        parameters.coverage_status = "both"

        # run baseline (to the very end)
        if step == 1:
            parameters.driver_cmd_line_args = ["parthenon/job/problem_id=gold"]
        elif step == 2:
            parameters.driver_cmd_line_args = [
                "-r",
                "gold.out0.00004.rhdf",
                "parthenon/job/problem_id=silver",
            ]
        elif step == 3:
            parameters.driver_cmd_line_args = [
                "-r",
                "gold.out0.00004.rhdf",
                "parthenon/job/problem_id=silver_coalesced",
                "parthenon/mesh/do_coalesced_comms=true",
            ]
        # check that we can dynamically enable outputs
        else:
            parameters.driver_cmd_line_args = [
                "-r",
                "gold.out0.00005.rhdf",
                "parthenon/job/problem_id=bronze",
                "parthenon/output1/file_type=hdf5",
                "parthenon/output1/dt=0.25",
                "parthenon/output1/last_time=0.25",
                "parthenon/output1/variables=advection.C, advection.phi, nodal.phi, advection.phi_fine_restricted, advection.phi_fine",
            ]
        return parameters

    def Analyse(self, parameters):
        sys.path.insert(
            1,
            parameters.parthenon_path
            + "/scripts/python/packages/parthenon_tools/parthenon_tools",
        )

        try:
            from phdf_diff import compare
        except ModuleNotFoundError:
            print("Couldn't find module to compare Parthenon hdf5 files.")
            return False

        success = True

        def compare_files(name, base="silver"):
            delta = compare(
                [
                    "gold.out0.%s.rhdf" % name,
                    "{}.out0.{}.rhdf".format(base, name),
                ],
                one=True,
                tol=0.0,
            )

            if delta != 0:
                print(
                    "ERROR: Found difference between gold and {} output {}.".format(
                        base, name
                    )
                )
                return False
            delta = compare(
                [
                    "bronze.out1.%s.phdf" % name,
                    "{}.out2.{}.phdf".format(base, name),
                ],
                # no need for metadata as the dynamically added output will cause
                # different metadata and we're just interested in the right data
                # being there.
                one=True,
                check_metadata=False,
            )

            if delta != 0:
                print(
                    "ERROR: Found difference between gold and bronze output '%s'."
                    % name
                )
                return False

            return True

        # compare a few files throughout the simulations
        success &= compare_files("final", "silver")
        success &= compare_files("final", "silver_coalesced")
        success &= check_xdmf_geometry("gold.out2.00000.phdf", expected_num_dims=2)

        return success
