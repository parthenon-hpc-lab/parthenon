//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020-2021. Triad National Security, LLC. All rights reserved.
//
// This program was produced under U.S. Government contract 89233218CNA000001 for Los
// Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
// for the U.S. Department of Energy/National Nuclear Security Administration. All rights
// in the program are reserved by Triad National Security, LLC, and the U.S. Department
// of Energy/National Nuclear Security Administration. The Government is granted for
// itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
// license in this material to reproduce, prepare derivative works, distribute copies to
// the public, perform publicly and display publicly, and to permit others to do so.
//========================================================================================

// Only proceed if HDF5 output enabled

#include <memory>

#include "outputs/parthenon_hdf5.hpp"

#include "mesh/meshblock.hpp"

#ifdef HDF5OUTPUT

namespace parthenon {

// XDMF subroutine to write a dataitem that refers to an HDF array
static std::string stringXdmfArrayRef(const std::string &prefix,
                                      const std::string &hdfPath,
                                      const std::string &label, const hsize_t *dims,
                                      const int &ndims, const std::string &theType,
                                      const int &precision) {
  std::string mystr =
      prefix + R"(<DataItem Format="HDF" Dimensions=")" + std::to_string(dims[0]);
  for (int i = 1; i < ndims; i++)
    mystr += " " + std::to_string(dims[i]);
  mystr += "\" Name=\"" + label + "\"";
  mystr += " NumberType=\"" + theType + "\"";
  mystr += R"( Precision=")" + std::to_string(precision) + R"(">)" + '\n';
  mystr += prefix + "  " + hdfPath + label + "</DataItem>" + '\n';
  return mystr;
}

static void writeXdmfArrayRef(std::ofstream &fid, const std::string &prefix,
                              const std::string &hdfPath, const std::string &label,
                              const hsize_t *dims, const int &ndims,
                              const std::string &theType, const int &precision) {
  fid << stringXdmfArrayRef(prefix, hdfPath, label, dims, ndims, theType, precision)
      << std::flush;
}

static void writeXdmfSlabVariableRef(std::ofstream &fid, std::string &name,
                                     std::string &hdfFile, int iblock, const int &vlen,
                                     int &ndims, hsize_t *dims,
                                     const std::string &dims321, bool isVector) {
  // writes a slab reference to file

  std::vector<std::string> names;
  int nentries = 1;
  int vector_size = 1;
  if (vlen == 1 || isVector) {
    names.push_back(name);
  } else {
    nentries = vlen;
    for (int i = 0; i < vlen; i++) {
      names.push_back(name + "_" + std::to_string(i));
    }
  }
  if (isVector) vector_size = vlen;

  const std::string prefix = "      ";
  for (int i = 0; i < nentries; i++) {
    fid << prefix << R"(<Attribute Name=")" << names[i] << R"(" Center="Cell")";
    if (isVector) {
      fid << R"( AttributeType="Vector")"
          << R"( Dimensions=")" << dims321 << " " << vector_size << R"(")";
    }
    fid << ">" << std::endl;
    fid << prefix << "  "
        << R"(<DataItem ItemType="HyperSlab" Dimensions=")" << dims321 << " "
        << vector_size << R"(">)" << std::endl;
    fid << prefix << "    "
        << R"(<DataItem Dimensions="3 5" NumberType="Int" Format="XML">)" << iblock
        << " 0 0 0 " << i << " 1 1 1 1 1 1 " << dims321 << " " << vector_size
        << "</DataItem>" << std::endl;
    writeXdmfArrayRef(fid, prefix + "    ", hdfFile + ":/", name, dims, ndims, "Float",
                      8);
    fid << prefix << "  "
        << "</DataItem>" << std::endl;
    fid << prefix << "</Attribute>" << std::endl;
  }
  return;
}

void PHDF5Output::genXDMF(std::string hdfFile, Mesh *pm, SimTime *tm) {
  // using round robin generation.
  // must switch to MPIIO at some point

  // only rank 0 writes XDMF
  if (Globals::my_rank != 0) {
    return;
  }
  std::string filename_aux = hdfFile + ".xdmf";
  std::ofstream xdmf;
  hsize_t dims[5] = {0, 0, 0, 0, 0};

  // open file
  xdmf = std::ofstream(filename_aux.c_str(), std::ofstream::trunc);

  // Write header
  xdmf << R"(<?xml version="1.0" ?>)" << std::endl;
  xdmf << R"(<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd">)" << std::endl;
  xdmf << R"(<Xdmf Version="3.0">)" << std::endl;
  xdmf << "  <Domain>" << std::endl;
  xdmf << R"(  <Grid Name="Mesh" GridType="Collection">)" << std::endl;
  if (tm != nullptr) {
    xdmf << R"(    <Time Value=")" << tm->time << R"("/>)" << std::endl;
    xdmf << R"(    <Information Name="Cycle" Value=")" << tm->ncycle << R"("/>)"
         << std::endl;
  }

  std::string blockTopology = R"(      <Topology Type="3DRectMesh" NumberOfElements=")" +
                              std::to_string(nx3 + 1) + " " + std::to_string(nx2 + 1) +
                              " " + std::to_string(nx1 + 1) + R"("/>)" + '\n';
  const std::string slabPreDim = R"(        <DataItem ItemType="HyperSlab" Dimensions=")";
  const std::string slabPreBlock2D =
      R"("><DataItem Dimensions="3 2" NumberType="Int" Format="XML">)";
  const std::string slabTrailer = "</DataItem>";

  // Now write Grid for each block
  dims[0] = pm->nbtotal;
  std::string dims321 =
      std::to_string(nx3) + " " + std::to_string(nx2) + " " + std::to_string(nx1);

  int ndims = 5;

  // same set of variables for all grids so use only one container
  auto ciX = MeshBlockDataIterator<Real>(pm->block_list.front()->meshblock_data.Get(),
                                         output_params.variables);
  for (int ib = 0; ib < pm->nbtotal; ib++) {
    xdmf << "    <Grid GridType=\"Uniform\" Name=\"" << ib << "\">" << std::endl;
    xdmf << blockTopology;
    xdmf << R"(      <Geometry Type="VXVYVZ">)" << std::endl;
    xdmf << slabPreDim << nx1 + 1 << slabPreBlock2D << ib << " 0 1 1 1 " << nx1 + 1
         << slabTrailer << std::endl;

    dims[1] = nx1 + 1;
    writeXdmfArrayRef(xdmf, "          ", hdfFile + ":/Locations/", "x", dims, 2, "Float",
                      8);
    xdmf << "</DataItem>" << std::endl;

    xdmf << slabPreDim << nx2 + 1 << slabPreBlock2D << ib << " 0 1 1 1 " << nx2 + 1
         << slabTrailer << std::endl;

    dims[1] = nx2 + 1;
    writeXdmfArrayRef(xdmf, "          ", hdfFile + ":/Locations/", "y", dims, 2, "Float",
                      8);
    xdmf << "</DataItem>" << std::endl;

    xdmf << slabPreDim << nx3 + 1 << slabPreBlock2D << ib << " 0 1 1 1 " << nx3 + 1
         << slabTrailer << std::endl;

    dims[1] = nx3 + 1;
    writeXdmfArrayRef(xdmf, "          ", hdfFile + ":/Locations/", "z", dims, 2, "Float",
                      8);
    xdmf << "</DataItem>" << std::endl;

    xdmf << "      </Geometry>" << std::endl;

    // write graphics variables
    dims[1] = nx3;
    dims[2] = nx2;
    dims[3] = nx1;
    dims[4] = 1;
    for (auto &v : ciX.vars) {
      const int vlen = v->GetDim(4);
      dims[4] = vlen;
      std::string name = v->label();
      writeXdmfSlabVariableRef(xdmf, name, hdfFile, ib, vlen, ndims, dims, dims321,
                               v->IsSet(Metadata::Vector));
    }
    xdmf << "      </Grid>" << std::endl;
  }
  xdmf << "    </Grid>" << std::endl;
  xdmf << "  </Domain>" << std::endl;
  xdmf << "</Xdmf>" << std::endl;
  xdmf.close();

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void PHDF5Output:::WriteOutputFile(Mesh *pm, ParameterInput *pin, bool flag)
//  \brief Cycles over all MeshBlocks and writes OutputData in the Parthenon HDF5 format,
//         one file per output using parallel IO.
void PHDF5Output::WriteOutputFile(Mesh *pm, ParameterInput *pin, SimTime *tm) {
  // writes all graphics variables to hdf file
  // HDF5 structures
  // Also writes companion xdmf file
  int max_blocks_global = pm->nbtotal;

  const IndexDomain theDomain =
      (output_params.include_ghost_zones ? IndexDomain::entire : IndexDomain::interior);

  auto const &first_block = *(pm->block_list.front());

  // shooting a blank just for getting the variable names
  IndexRange out_ib = first_block.cellbounds.GetBoundsI(theDomain);
  IndexRange out_jb = first_block.cellbounds.GetBoundsJ(theDomain);
  IndexRange out_kb = first_block.cellbounds.GetBoundsK(theDomain);

  int const num_blocks_local = static_cast<int>(pm->block_list.size());

  auto nblist = pm->GetNbList();

  auto ciX = MeshBlockDataIterator<Real>(pm->block_list.front()->meshblock_data.Get(),
                                         output_params.variables);

  // set output size
  nx1 = out_ib.e - out_ib.s + 1; // SS first_block.block_size.nx1;
  nx2 = out_jb.e - out_jb.s + 1; // SS first_block.block_size.nx2;
  nx3 = out_kb.e - out_kb.s + 1; // SS first_block.block_size.nx3;

  // open HDF5 file
  // Define output filename
  filename = std::string(output_params.file_basename);
  filename.append(".");
  filename.append(output_params.file_id);
  filename.append(".");
  std::stringstream file_number;
  file_number << std::setw(5) << std::setfill('0') << output_params.file_number;
  filename.append(file_number.str());
  filename.append(".phdf");
  {
#ifdef MPI_PARALLEL
    /* set the file access template for parallel IO access */
    H5P const acc_file = H5P::FromHIDCheck(H5Pcreate(H5P_FILE_ACCESS));

    /* ---------------------------------------------------------------------
       platform dependent code goes here -- the access template must be
       tuned for a particular filesystem blocksize.  some of these
       numbers are guesses / experiments, others come from the file system
       documentation.

       The sieve_buf_size should be equal a multiple of the disk block size
       ---------------------------------------------------------------------- */

    /* create an MPI_INFO object -- on some platforms it is useful to
       pass some information onto the underlying MPI_File_open call */
    MPI_Info FILE_INFO_TEMPLATE;
    PARTHENON_MPI_CHECK(MPI_Info_create(&FILE_INFO_TEMPLATE));

    // Free MPI_Info on error on return or throw
    struct MPI_InfoDeleter {
      MPI_Info info;
      ~MPI_InfoDeleter() { MPI_Info_free(&info); }
    } delete_info{FILE_INFO_TEMPLATE};

    PARTHENON_HDF5_CHECK(H5Pset_sieve_buf_size(acc_file, 262144));
    PARTHENON_HDF5_CHECK(H5Pset_alignment(acc_file, 524288, 262144));

    PARTHENON_MPI_CHECK(MPI_Info_set(FILE_INFO_TEMPLATE, "access_style", "write_once"));
    PARTHENON_MPI_CHECK(MPI_Info_set(FILE_INFO_TEMPLATE, "collective_buffering", "true"));
    PARTHENON_MPI_CHECK(MPI_Info_set(FILE_INFO_TEMPLATE, "cb_block_size", "1048576"));
    PARTHENON_MPI_CHECK(MPI_Info_set(FILE_INFO_TEMPLATE, "cb_buffer_size", "4194304"));

    /* tell the HDF5 library that we want to use MPI-IO to do the writing */
    PARTHENON_HDF5_CHECK(H5Pset_fapl_mpio(acc_file, MPI_COMM_WORLD, FILE_INFO_TEMPLATE));
    PARTHENON_HDF5_CHECK(H5Pset_fapl_mpio(acc_file, MPI_COMM_WORLD, MPI_INFO_NULL));
#else
    hid_t const acc_file = H5P_DEFAULT;
#endif

    // now open the file
    H5F const file = H5F::FromHIDCheck(
        H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, acc_file));

    // write timestep relevant attributes
    { // START /Info
      // attributes written here:
      // All ranks write attributes
      H5S const localDSpace = H5S::FromHIDCheck(H5Screate(H5S_SCALAR));
      H5D const infoDSet = H5D::FromHIDCheck(H5Dcreate(
          file, "/Info", PREDINT32, localDSpace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));

      int max_level = pm->GetCurrentLevel() - pm->GetRootLevel();
      if (tm != nullptr) {
        writeH5AI32("NCycle", &(tm->ncycle), localDSpace, infoDSet);
        writeH5AF64("Time", &(tm->time), localDSpace, infoDSet);
      }
      writeH5AI32("NumDims", &pm->ndim, localDSpace, infoDSet);
      writeH5AI32("NumMeshBlocks", &pm->nbtotal, localDSpace, infoDSet);
      writeH5AI32("MaxLevel", &max_level, localDSpace, infoDSet);
      // write whether we include ghost cells or not
      int iTmp = (output_params.include_ghost_zones ? 1 : 0);
      writeH5AI32("IncludesGhost", &iTmp, localDSpace, infoDSet);
      // write number of ghost cells in simulation
      iTmp = Globals::nghost;
      writeH5AI32("NGhost", &iTmp, localDSpace, infoDSet);
      writeH5ASTRING("Coordinates", std::string(first_block.coords.Name()), localDSpace,
                     infoDSet);

      hsize_t nPE = Globals::nranks;
      writeH5AI32("BlocksPerPE", nblist.data(),
                  H5S::FromHIDCheck(H5Screate_simple(1, &nPE, NULL)), infoDSet);

      // open vector space
      // write mesh block size
      const hsize_t xDims[] = {3};
      int meshblock_size[] = {nx1, nx2, nx3};
      writeH5AI32("MeshBlockSize", meshblock_size,
                  H5S::FromHIDCheck(H5Screate_simple(1, xDims, NULL)), infoDSet);

      // RootGridDomain - float[9] array with xyz mins, maxs, rats (dx(i)/dx(i-1))
      const hsize_t rootGridDomain_size[] = {9};
      Real rootGridDomain[] = {
          pm->mesh_size.x1min, pm->mesh_size.x1max, pm->mesh_size.x1rat,
          pm->mesh_size.x2min, pm->mesh_size.x2max, pm->mesh_size.x2rat,
          pm->mesh_size.x3min, pm->mesh_size.x3max, pm->mesh_size.x3rat};
      writeH5AF64("RootGridDomain", rootGridDomain,
                  H5S::FromHIDCheck(H5Screate_simple(1, rootGridDomain_size, NULL)),
                  infoDSet);

      // RootGridSize - int[3] number of cells on the root grid
      const hsize_t rootGridSize_size[] = {3};
      int rootGridSize[] = {pm->mesh_size.nx1, pm->mesh_size.nx2, pm->mesh_size.nx3};
      writeH5AI32("RootGridSize", rootGridSize,
                  H5S::FromHIDCheck(H5Screate_simple(1, rootGridSize_size, NULL)),
                  infoDSet);

      // BoundaryConditions
      const hsize_t boundaryConditions_size[1] = {BOUNDARY_NFACES};
      std::vector<std::string> boundaryConditions;
      boundaryConditions.reserve(boundaryConditions_size[0]);
      for (int i = 0; i < BOUNDARY_NFACES; i++) {
        boundaryConditions.push_back(GetBoundaryString(pm->mesh_bcs[i]));
      }
      writeH5ASTRINGS(
          "BoundaryConditions", boundaryConditions,
          H5S::FromHIDCheck(H5Screate_simple(1, boundaryConditions_size, NULL)),
          infoDSet);

      // DatasetNames - which datasets from the top level to read
      const hsize_t datasetNames_size[1] = {output_params.variables.size()};
      writeH5ASTRINGS("DatasetNames", output_params.variables,
                      H5S::FromHIDCheck(H5Screate_simple(1, datasetNames_size, NULL)),
                      infoDSet);

      // NumVariables - number of variables within each dataset
      std::vector<int> numVariables(output_params.variables.size());
      // VariablesNames - Names of variables within each dataset
      std::vector<std::string> variableNames;

      for (int i = 0; i < ciX.vars.size(); i++) { // for every block
        const std::vector<std::string> component_labels =
            ciX.vars[i]->metadata().getComponentLabels();

        if (component_labels.size() > 0) {
          numVariables[i] = component_labels.size();
          for (const auto &label : component_labels) {
            variableNames.push_back(label);
          }
        } else {
          numVariables[i] = 1;
          variableNames.push_back(ciX.vars[i]->label());
        }
      }

      writeH5AI32("NumVariables", numVariables.data(),
                  H5S::FromHIDCheck(H5Screate_simple(1, datasetNames_size, NULL)),
                  infoDSet);

      if (variableNames.size() > 0) {
        const hsize_t variableNames_size[1] = {variableNames.size()};
        writeH5ASTRINGS("VariableNames", variableNames,
                        H5S::FromHIDCheck(H5Screate_simple(1, variableNames_size, NULL)),
                        infoDSet);
      }
    } // END /Info

    { // START /Params
      // Params written here:
      // All ranks write attributes
      H5S const localDSpace = H5S::FromHIDCheck(H5Screate(H5S_SCALAR));
      H5D const infoDSet =
          H5D::FromHIDCheck(H5Dcreate(file, "/Params", PREDINT32, localDSpace,
                                      H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));

      for (const auto &package : pm->packages.AllPackages()) {
        std::shared_ptr<StateDescriptor> state = package.second;
        // Write AllParams that can be written as HDF5 attributes
        std::vector<std::string> paramKeys = state->AllParams().GetKeys();
        for (const auto &paramKey : paramKeys) {
          const std::type_index type = state->ParamType(paramKey);

          std::stringstream ss;
          ss << state->label() << "/" << paramKey;

          if (type == typeid(int)) {
            const int &param = state->Param<int>(paramKey);
            writeH5AI32(ss.str().c_str(), &param, localDSpace, infoDSet);
          } else if (type == typeid(double)) {
            const double &param = state->Param<double>(paramKey);
            writeH5AF64(ss.str().c_str(), &param, localDSpace, infoDSet);
          } else if (type == typeid(std::string)) {
            const std::string &param = state->Param<std::string>(paramKey);
            writeH5ASTRING(ss.str().c_str(), param, localDSpace, infoDSet);
          }
        }
      }
    } // END /Params

    // allocate space for largest size variable
    size_t maxV = 1;
    hsize_t sumDim4AllVars = 0;
    for (auto &v : ciX.vars) {
      const size_t vlen = v->GetDim(4);
      if (!v->metadata().IsSet(Metadata::None)) {
        sumDim4AllVars += vlen;
        maxV = (maxV < vlen ? vlen : maxV);
      }
    }

    std::vector<Real> tmpData((nx1 + 1) * (nx2 + 1) * (nx3 + 1) * maxV *
                              num_blocks_local);
    for (int i = 0; i < (nx1 + 1) * (nx2 + 1) * (nx3 + 1) * maxV * num_blocks_local; i++)
      tmpData[i] = -1.25;

    // Write mesh coordinates to file
    hsize_t local_start[7], global_count[7], local_count[7];
    for (int i = 0; i < 7; i++) {
      local_start[i] = 0;
    }

    // Shift local starting block
    for (int i = 0; i < Globals::my_rank; i++) {
      local_start[0] += nblist[i];
    }
    H5P const property_list = H5P::FromHIDCheck(H5Pcreate(H5P_DATASET_XFER));
#ifdef MPI_PARALLEL
    PARTHENON_HDF5_CHECK(H5Pset_dxpl_mpio(property_list, H5FD_MPIO_COLLECTIVE));
#endif

    // set starting poing in hyperslab for our blocks and
    // number of blocks on our PE
    {
      // open locations tab
      H5G const gLocations = H5G::FromHIDCheck(
          H5Gcreate(file, "/Locations", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));

      // write X coordinates
      local_count[0] = num_blocks_local;
      global_count[0] = max_blocks_global;

      // These macros are defined in parthenon_hdf5.hpp, which establishes relevant scope
      LOADVARIABLEALL(tmpData, pm, pmb->coords.x1f, out_ib.s, out_ib.e + 1, 0, 0, 0, 0);
      local_count[1] = global_count[1] = nx1 + 1;
      WRITEH5SLAB("x", tmpData.data(), gLocations, local_start, local_count, global_count,
                  property_list);

      // write Y coordinates
      LOADVARIABLEALL(tmpData, pm, pmb->coords.x2f, 0, 0, out_jb.s, out_jb.e + 1, 0, 0);
      local_count[1] = global_count[1] = nx2 + 1;
      WRITEH5SLAB("y", tmpData.data(), gLocations, local_start, local_count, global_count,
                  property_list);

      // write Z coordinates
      LOADVARIABLEALL(tmpData, pm, pmb->coords.x3f, 0, 0, 0, 0, out_kb.s, out_kb.e + 1);

      local_count[1] = global_count[1] = nx3 + 1;
      WRITEH5SLAB("z", tmpData.data(), gLocations, local_start, local_count, global_count,
                  property_list);
    }
    {
      // open locations tab
      H5G const gVLocations = H5G::FromHIDCheck(
          H5Gcreate(file, "/VolumeLocations", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));

      // write X coordinates
      local_count[0] = num_blocks_local;
      global_count[0] = max_blocks_global;

      // These macros are defined in parthenon_hdf5.hpp, which establishes relevant scope
      LOADVARIABLEALL(tmpData, pm, pmb->coords.x1v, out_ib.s, out_ib.e, 0, 0, 0, 0);
      local_count[1] = global_count[1] = nx1;
      WRITEH5SLAB("x", tmpData.data(), gVLocations, local_start, local_count,
                  global_count, property_list);

      // write Y coordinates
      LOADVARIABLEALL(tmpData, pm, pmb->coords.x2v, 0, 0, out_jb.s, out_jb.e, 0, 0);
      local_count[1] = global_count[1] = nx2;
      WRITEH5SLAB("y", tmpData.data(), gVLocations, local_start, local_count,
                  global_count, property_list);

      // write Z coordinates
      LOADVARIABLEALL(tmpData, pm, pmb->coords.x3v, 0, 0, 0, 0, out_kb.s, out_kb.e);

      local_count[1] = global_count[1] = nx3;
      WRITEH5SLAB("z", tmpData.data(), gVLocations, local_start, local_count,
                  global_count, property_list);
    }

    {
      // Write Levels and Logical Locations with the level for each Meshblock
      // loclist contains levels and logical locations for all meshblocks on
      // all ranks
      const auto &loclist = pm->GetLocList();

      std::vector<std::int64_t> levels;
      levels.reserve(pm->nbtotal);

      std::vector<std::int64_t> logicalLocations;
      logicalLocations.reserve(pm->nbtotal * 3);

      for (const auto &loc : loclist) { // for every block
        levels.push_back(loc.level - pm->GetRootLevel());
        logicalLocations.push_back(loc.lx1);
        logicalLocations.push_back(loc.lx2);
        logicalLocations.push_back(loc.lx3);
      }

      // Only write levels on rank 0
      local_count[0] = (Globals::my_rank == 0) ? pm->nbtotal : 0;

      global_count[0] = max_blocks_global;

      H5S const local_levelsSpace =
          H5S::FromHIDCheck(H5Screate_simple(1, local_count, NULL));
      H5S const global_levelsSpace =
          H5S::FromHIDCheck(H5Screate_simple(1, global_count, NULL));

      WRITEH5SLAB_X("Levels", levels.data(), file, local_start, local_count,
                    local_levelsSpace, global_levelsSpace, property_list,
                    H5T_NATIVE_INT32);

      // Only write LogicalLocations on rank 0
      local_count[0] = (Globals::my_rank == 0) ? pm->nbtotal : 0;
      local_count[1] = 3;

      global_count[0] = max_blocks_global;
      global_count[1] = 3;

      WRITEH5SLABI64("LogicalLocations", logicalLocations.data(), file, local_start,
                     local_count, global_count, property_list);
    }

    // write variables
    // create persistent spaces
    local_count[0] = num_blocks_local;
    local_count[1] = nx3;
    local_count[2] = nx2;
    local_count[3] = nx1;
    local_count[4] = 1;

    global_count[0] = max_blocks_global;
    global_count[1] = nx3;
    global_count[2] = nx2;
    global_count[3] = nx1;
    global_count[4] = 1;

    H5S const local_DSpace = H5S::FromHIDCheck(H5Screate_simple(5, local_count, NULL));
    H5S const global_DSpace = H5S::FromHIDCheck(H5Screate_simple(5, global_count, NULL));

    // while we could do this as n variables and load all variables for
    // a block at one time, this is memory-expensive.  I think it is
    // well worth the multiple iterations through the blocks to load up
    // one variable at a time.  Besides most of the time will be spent
    // writing the HDF5 file to disk anyway...
    // If I'm wrong about this, we can always rewrite this later.
    // Sriram

    // this is a stupidly complicated multi-pass through the variable
    // list, but again will revisit when the time comes to redo
    for (auto &vwrite : ciX.vars) { // for each variable we write
      const std::string vWriteName = vwrite->label();
      hid_t vLocalSpace, vGlobalSpace;
      H5S vLocalSpaceNew, vGlobalSpaceNew;
      const hsize_t vlen = vwrite->GetDim(4);
      local_count[4] = global_count[4] = vlen;

      if (vwrite->metadata().IsSet(Metadata::None)) {
        continue;
      }
      if (vlen == 1) {
        vLocalSpace = local_DSpace;
        vGlobalSpace = global_DSpace;
      } else {
        vLocalSpace = vLocalSpaceNew =
            H5S::FromHIDCheck(H5Screate_simple(5, local_count, NULL));
        vGlobalSpace = vGlobalSpaceNew =
            H5S::FromHIDCheck(H5Screate_simple(5, global_count, NULL));
      }

      hsize_t index = 0;
      for (auto &pmb : pm->block_list) { // for every block1
        auto ci = MeshBlockDataIterator<Real>(pmb->meshblock_data.Get(),
                                              output_params.variables);
        for (auto &v : ci.vars) {
          std::string name = v->label();
          if (name.compare(vWriteName) == 0) {
            // hsize_t index = pmb->lid * varSize * vlen;
            auto v_h = v->data.GetHostMirrorAndCopy();
            LOADVARIABLEONE(index, tmpData, v_h, out_ib.s, out_ib.e, out_jb.s, out_jb.e,
                            out_kb.s, out_kb.e, vlen);
            break;
          }
        }
      }
      // write dataset to file
      WRITEH5SLAB2(vWriteName.c_str(), tmpData.data(), file, local_start, local_count,
                   vLocalSpace, vGlobalSpace, property_list);
    }

    // Block variables
    for (auto &vwrite : ciX.vars) { // for each variable we write
      if (!vwrite->metadata().IsSet(Metadata::None)) {
        continue;
      }
      const std::string vWriteName = vwrite->label();
      hid_t vLocalSpace, vGlobalSpace;
      H5S vLocalSpaceNew, vGlobalSpaceNew;

      // Count up dimensions to output
      int spaceCount;
      spaceCount = 1;
      for (int idim = 1; idim <= 6; idim++) {
        if (vwrite->GetDim(idim) > 1) {
          local_count[spaceCount] = global_count[spaceCount] = vwrite->GetDim(idim);
          spaceCount++;

          // copy rest verbatim into the counts
          for (int jdim = idim + 1; jdim <= 6; jdim++) {
            local_count[spaceCount] = global_count[spaceCount] = vwrite->GetDim(jdim);
            spaceCount++;
          }
          break;
        }
      }
      vLocalSpace = vLocalSpaceNew =
          H5S::FromHIDCheck(H5Screate_simple(spaceCount, local_count, NULL));
      vGlobalSpace = vGlobalSpaceNew =
          H5S::FromHIDCheck(H5Screate_simple(spaceCount, global_count, NULL));

      hsize_t index = 0;
      for (auto &pmb : pm->block_list) { // for every block1
        auto ci = MeshBlockDataIterator<Real>(pmb->meshblock_data.Get(),
                                              output_params.variables);
        for (auto &v : ci.vars) {
          std::string name = v->label();
          if (name.compare(vWriteName) == 0) {
            auto v_h = v->data.GetHostMirrorAndCopy();
            LOADVARIABLERAW(index, tmpData, v_h, v->GetDim(1), v->GetDim(2), v->GetDim(3),
                            v->GetDim(4), v->GetDim(5), v->GetDim(6));
            break;
          }
        }
      }
      // write dataset to file
      WRITEH5SLAB2(vWriteName.c_str(), tmpData.data(), file, local_start, local_count,
                   vLocalSpace, vGlobalSpace, property_list);
    }
  }

  // generate XDMF companion file
  (void)genXDMF(filename, pm, tm);

  // advance output parameters
  output_params.file_number++;
  output_params.next_time += output_params.dt;
  pin->SetInteger(output_params.block_name, "file_number", output_params.file_number);
  pin->SetReal(output_params.block_name, "next_time", output_params.next_time);
  return;
}

} // namespace parthenon

#endif // HDF5OUTPUT
