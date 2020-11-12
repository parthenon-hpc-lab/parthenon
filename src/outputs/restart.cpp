//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020. Triad National Security, LLC. All rights reserved.
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
//! \file restart.cpp
//  \brief writes restart files

#include <memory>
#include <string>
#include <utility>

#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "outputs/outputs.hpp"
#include "outputs/restart.hpp"

namespace parthenon {

//----------------------------------------------------------------------------------------
//! \fn void RestartReader::RestartReader(const std::string filename)
//  \brief Opens the restart file and stores appropriate file handle in fh_
RestartReader::RestartReader(const char *filename) : filename_(filename) {
#ifndef HDF5OUTPUT
  std::stringstream msg;
  msg << "### FATAL ERROR in Restart (Reader) constructor" << std::endl
      << "Executable not configured for HDF5 outputs, but HDF5 file format "
      << "is required for restarts" << std::endl;
  PARTHENON_FAIL(msg);
#else
  // Open the HDF file in read only mode
  fh_ = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);

  // populate block size from the file
  std::vector<int32_t> blockSize = ReadAttr1DI32("Mesh", "blockSize");
  hasGhost = GetAttr<int>("Mesh", "includesGhost");
  nx1_ = static_cast<hsize_t>(blockSize[0]);
  nx2_ = static_cast<hsize_t>(blockSize[1]);
  nx3_ = static_cast<hsize_t>(blockSize[2]);
#endif
}

//! \fn std::shared_ptr<std::vector<T>> RestartReader::ReadAttrString(const char *dataset,
//! const char *name, size_t *count = nullptr)
//  \brief Reads a string attribute for given dataset
std::string RestartReader::ReadAttrString(const char *dataset, const char *name,
                                          size_t *count) {
  // Returns entire 1D array.
  // status, never checked.  We should...
#ifdef HDF5OUTPUT
  herr_t status;

  hid_t theHdfType = H5T_C_S1;

  hid_t dset = H5Dopen2(fh_, dataset, H5P_DEFAULT);
  hid_t attr = H5Aopen(dset, name, H5P_DEFAULT);
  hid_t dataspace = H5Aget_space(attr);

  // Allocate array of correct size
  hid_t filetype = H5Aget_type(attr);
  hsize_t isize = H5Tget_size(filetype);
  isize++;
  if (count != nullptr) {
    *count = isize;
  }

  char *s = static_cast<char *>(calloc(isize + 1, sizeof(char)));
  // Read data from file
  //  status = H5Aread(attr, theHdfType, static_cast<void *>(s));
  hid_t memType = H5Tcopy(H5T_C_S1);
  status = H5Tset_size(memType, isize);
  status = H5Aread(attr, memType, s);
  std::string data(s);
  free(s);

  // CLose the dataspace and data set.
  H5Tclose(memType);
  H5Tclose(filetype);
  H5Sclose(dataspace);
  H5Aclose(attr);
  H5Dclose(dset);

  return data;
#else
  return std::string("HDF5 NOT COMPILED IN");
#endif
}

//----------------------------------------------------------------------------------------
//! \fn void RestartOutput::WriteOutputFile(Mesh *pm, ParameterInput *pin, bool flag)
//  \brief Cycles over all MeshBlocks and writes data to a single restart file.
void RestartOutput::WriteOutputFile(Mesh *pm, ParameterInput *pin, SimTime *tm) {
  // Restart output is currently only HDF5, so no HDF5 means no restart files
#ifdef HDF5OUTPUT
  // Writes a restart file in 'rhdf' format
  // This format has:
  //   /Input: Current input parameter key-value pairs
  //   /Info: information about simulation
  //   /Mesh: Information on mesh
  //   /Blocks: Metadata for blocks
  //   /var1: variable data
  //   /var2: variable data
  //   ....
  //   /varN: variable data
  //
  // It is expected that on restart global block ID will determine which data set is
  // read.
  //
  hsize_t max_blocks_global = pm->nbtotal;
  hsize_t num_blocks_local = 0;

  // SSconst IndexDomain interior = IndexDomain::interior;
  int iGhost = (output_params.include_ghost_zones ? 1 : 0);

  const IndexDomain theDomain = (iGhost ? IndexDomain::entire : IndexDomain::interior);

  auto &mb = *(pm->block_list.front());

  // shooting a blank just for getting the variable names
  const IndexRange out_ib = mb.cellbounds.GetBoundsI(theDomain);
  const IndexRange out_jb = mb.cellbounds.GetBoundsJ(theDomain);
  const IndexRange out_kb = mb.cellbounds.GetBoundsK(theDomain);

  // Should this just be pm->block_list.size()?
  for (auto &pmb : pm->block_list) {
    num_blocks_local++;
  }
  // set output size

  // open HDF5 file
  // Define output filename
  auto filename = std::string(output_params.file_basename);
  filename.append(".");
  filename.append(output_params.file_id);
  filename.append(".");
  std::stringstream file_number;
  file_number << std::setw(5) << std::setfill('0') << output_params.file_number;
  filename.append(file_number.str());
  filename.append(".rhdf");

  hid_t file;
  hid_t acc_file = H5P_DEFAULT;

#ifdef MPI_PARALLEL
  /* set the file access template for parallel IO access */
  acc_file = H5Pcreate(H5P_FILE_ACCESS);

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
  int ierr;
  MPI_Status stat;
  ierr = MPI_Info_create(&FILE_INFO_TEMPLATE);
  ierr = H5Pset_sieve_buf_size(acc_file, 262144);
  ierr = H5Pset_alignment(acc_file, 524288, 262144);

  ierr = MPI_Info_set(FILE_INFO_TEMPLATE, "access_style", "write_once");
  ierr = MPI_Info_set(FILE_INFO_TEMPLATE, "collective_buffering", "true");
  ierr = MPI_Info_set(FILE_INFO_TEMPLATE, "cb_block_size", "1048576");
  ierr = MPI_Info_set(FILE_INFO_TEMPLATE, "cb_buffer_size", "4194304");

  /* tell the HDF5 library that we want to use MPI-IO to do the writing */
  ierr = H5Pset_fapl_mpio(acc_file, MPI_COMM_WORLD, FILE_INFO_TEMPLATE);
  ierr = H5Pset_fapl_mpio(acc_file, MPI_COMM_WORLD, MPI_INFO_NULL);
#endif

  // now open the file
  file = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, acc_file);

  // attributes written here:
  // All ranks write attributes

  // write timestep relevant attributes
  hid_t localDSpace, localnDSpace, myDSet;
  herr_t status;
  hsize_t nLen;

  { // write input key-value pairs
    std::ostringstream oss;
    pin->ParameterDump(oss);

    // Mesh information
    localDSpace = H5Screate(H5S_SCALAR);
    myDSet = H5Dcreate(file, "/Input", PREDINT32, localDSpace, H5P_DEFAULT, H5P_DEFAULT,
                       H5P_DEFAULT);

    status = writeH5ASTRING("File", oss.str(), file, localDSpace, myDSet);

    // close space and set
    status = H5Sclose(localDSpace);
    status = H5Dclose(myDSet);
  }

  localDSpace = H5Screate(H5S_SCALAR);
  myDSet = H5Dcreate(file, "/Info", PREDINT32, localDSpace, H5P_DEFAULT, H5P_DEFAULT,
                     H5P_DEFAULT);

  int rootLevel = pm->GetRootLevel();
  int max_level = pm->GetCurrentLevel() - rootLevel;
  if (tm != nullptr) {
    status = writeH5AI32("NCycle", &(tm->ncycle), file, localDSpace, myDSet);
    status = writeH5AF64("Time", &(tm->time), file, localDSpace, myDSet);
    status = writeH5AF64("dt", &(tm->dt), file, localDSpace, myDSet);
  }
  status = writeH5ASTRING("Coordinates", std::string(mb.coords.Name()), file, localDSpace,
                          myDSet);

  status = writeH5AI32("NumDims", &pm->ndim, file, localDSpace, myDSet);

  status = H5Sclose(localDSpace);

  hsize_t nPE = Globals::nranks;
  localDSpace = H5Screate_simple(1, &nPE, NULL);
  auto nblist = pm->GetNbList();
  status = writeH5AI32("BlocksPerPE", nblist.data(), file, localDSpace, myDSet);
  status = H5Sclose(localDSpace);
  status = H5Dclose(myDSet);

  // Mesh information
  localDSpace = H5Screate(H5S_SCALAR);
  myDSet = H5Dcreate(file, "/Mesh", PREDINT32, localDSpace, H5P_DEFAULT, H5P_DEFAULT,
                     H5P_DEFAULT);

  auto nx1 = out_ib.e - out_ib.s + 1; // SS mb.block_size.nx1;
  auto nx2 = out_jb.e - out_jb.s + 1; // SS mb.block_size.nx2;
  auto nx3 = out_kb.e - out_kb.s + 1; // SS mb.block_size.nx3;
  int bsize[3] = {mb.block_size.nx1, mb.block_size.nx2, mb.block_size.nx3};
  nLen = 3;
  localnDSpace = H5Screate_simple(1, &nLen, NULL);
  status = writeH5AI32("blockSize", bsize, file, localnDSpace, myDSet);
  status = writeH5AI32("includesGhost", &iGhost, file, localDSpace, myDSet);
  status = H5Sclose(localnDSpace);

  status = writeH5AI32("nbtotal", &pm->nbtotal, file, localDSpace, myDSet);
  status = writeH5AI32("nbnew", &pm->nbnew, file, localDSpace, myDSet);
  status = writeH5AI32("nbdel", &pm->nbdel, file, localDSpace, myDSet);
  status = writeH5AI32("rootLevel", &rootLevel, file, localDSpace, myDSet);
  status = writeH5AI32("MaxLevel", &max_level, file, localDSpace, myDSet);

  { // refinement flag
    int refine = (pm->adaptive ? 1 : 0);
    status = writeH5AI32("refine", &refine, file, localDSpace, myDSet);

    int multilevel = (pm->multilevel ? 1 : 0);
    status = writeH5AI32("multilevel", &multilevel, file, localDSpace, myDSet);
  }

  { // mesh bounds
    const auto &rs = pm->mesh_size;
    const Real limits[6] = {rs.x1min, rs.x2min, rs.x3min, rs.x1max, rs.x2max, rs.x3max};
    const Real ratios[3] = {rs.x1rat, rs.x2rat, rs.x3rat};
    nLen = 6;
    localnDSpace = H5Screate_simple(1, &nLen, NULL);
    status = writeH5AF64("bounds", limits, file, localnDSpace, myDSet);
    status = H5Sclose(localnDSpace);

    nLen = 3;
    localnDSpace = H5Screate_simple(1, &nLen, NULL);
    status = writeH5AF64("ratios", ratios, file, localnDSpace, myDSet);
    status = H5Sclose(localnDSpace);
  }

  { // boundary conditions
    nLen = 6;
    localnDSpace = H5Screate_simple(1, &nLen, NULL);
    int bcsi[6];
    for (int ib = 0; ib < 6; ib++) {
      bcsi[ib] = static_cast<int>(pm->mesh_bcs[ib]);
    }
    status = writeH5AI32("bc", bcsi, file, localnDSpace, myDSet);
    status = H5Sclose(localnDSpace);
  }

  // close space and set
  status = H5Sclose(localDSpace);
  status = H5Dclose(myDSet);

  // end mesh section

  // write blocks
  // MeshBlock information
  // Write mesh coordinates to file
  hsize_t local_start[5], global_count[5], local_count[5];
  hid_t gLocations;

  local_start[0] = 0;
  local_start[1] = 0;
  local_start[2] = 0;
  local_start[3] = 0;
  local_start[4] = 0;
  for (int i = 0; i < Globals::my_rank; i++) {
    local_start[0] += nblist[i];
  }
  hid_t property_list = H5Pcreate(H5P_DATASET_XFER);
#ifdef MPI_PARALLEL
  H5Pset_dxpl_mpio(property_list, H5FD_MPIO_COLLECTIVE);
#endif

  // set starting poing in hyperslab for our blocks and
  // number of blocks on our PE

  // open blocks tab
  hid_t gBlocks = H5Gcreate(file, "/Blocks", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  // write Xmin[ndim] for blocks
  {
    std::vector<Real> tmpData(num_blocks_local * 3);
    local_count[0] = num_blocks_local;
    global_count[0] = max_blocks_global;
    int i = 0;
    for (auto &pmb : pm->block_list) {
      auto xmin = pmb->coords.GetXmin();
      tmpData[i] = xmin[0];
      i++;
      if (pm->ndim > 1) {
        tmpData[i] = xmin[1];
        i++;
      }
      if (pm->ndim > 2) {
        tmpData[i] = xmin[2];
        i++;
      }
    }
    local_count[1] = global_count[1] = pm->ndim;
    WRITEH5SLABDOUBLE("xmin", tmpData.data(), gBlocks, local_start, local_count,
                      global_count, property_list);
  }

  // write Block ID
  {
    // LOC.lx1,2,3
    hsize_t n;
    int i;

    n = 3;
    std::vector<int64_t> tmpLoc(num_blocks_local * n);
    local_count[1] = global_count[1] = n;
    local_count[0] = num_blocks_local;
    global_count[0] = max_blocks_global;
    i = 0;
    for (auto &pmb : pm->block_list) {
      tmpLoc[i++] = pmb->loc.lx1;
      tmpLoc[i++] = pmb->loc.lx2;
      tmpLoc[i++] = pmb->loc.lx3;
    }
    WRITEH5SLABI64("loc.lx123", tmpLoc.data(), gBlocks, local_start, local_count,
                   global_count, property_list);

    // (LOC.)level, GID, LID, cnghost, gflag
    n = 5;
    std::vector<int> tmpID(num_blocks_local * n);
    local_count[1] = global_count[1] = n;
    local_count[0] = num_blocks_local;
    global_count[0] = max_blocks_global;
    i = 0;
    for (auto &pmb : pm->block_list) {
      tmpID[i++] = pmb->loc.level;
      tmpID[i++] = pmb->gid;
      tmpID[i++] = pmb->lid;
      tmpID[i++] = pmb->cnghost;
      tmpID[i++] = pmb->gflag;
    }
    WRITEH5SLABI32("loc.level-gid-lid-cnghost-gflag", tmpID.data(), gBlocks, local_start,
                   local_count, global_count, property_list);
  }

  // close locations tab
  H5Gclose(gBlocks);

  // write variables

  // write variables
  // create persistent spaces
  local_count[1] = nx3;
  local_count[2] = nx2;
  local_count[3] = nx1;
  local_count[4] = 1;

  global_count[1] = nx3;
  global_count[2] = nx2;
  global_count[3] = nx1;
  global_count[4] = 1;

  hid_t local_DSpace = H5Screate_simple(5, local_count, NULL);
  hid_t global_DSpace = H5Screate_simple(5, global_count, NULL);

  // while we could do this as n variables and load all variables for
  // a block at one time, this is memory-expensive.  I think it is
  // well worth the multiple iterations through the blocks to load up
  // one variable at a time.  Besides most of the time will be spent
  // writing the HDF5 file to disk anyway...
  // If I'm wrong about this, we can always rewrite this later.
  // Sriram

  const hsize_t varSize = nx3 * nx2 * nx1;

  auto ciX = MeshBlockDataIterator<Real>(
      mb.meshblock_data.Get(),
      {parthenon::Metadata::Independent, parthenon::Metadata::Restart}, true);
  for (auto &vwrite : ciX.varsCell) { // for each variable we write
    const std::string vWriteName = vwrite->label();
    hid_t vLocalSpace, vGlobalSpace;
    auto &mb = *(pm->block_list.front());
    const hsize_t vlen = vwrite->GetDim(4);
    local_count[4] = global_count[4] = vlen;
    std::vector<Real> tmpData(varSize * vlen * num_blocks_local);

    // create spaces if required
    if (vlen == 1) {
      vLocalSpace = local_DSpace;
      vGlobalSpace = global_DSpace;
    } else {
      vLocalSpace = H5Screate_simple(5, local_count, NULL);
      vGlobalSpace = H5Screate_simple(5, global_count, NULL);
    }

    // load up data
    hsize_t index = 0;
    for (auto &pmb : pm->block_list) {
      bool found = false;
      auto ci = MeshBlockDataIterator<Real>(
          pmb->meshblock_data.Get(),
          {parthenon::Metadata::Independent, parthenon::Metadata::Restart}, true);
      for (auto &v : ci.varsCell) {
        // Note index 4 transposed to interior
        if (vWriteName.compare(v->label()) == 0) {
          auto v_h = v->data.GetHostMirrorAndCopy();
          LOADVARIABLEONE(index, tmpData.data(), v_h, out_ib.s, out_ib.e, out_jb.s,
                          out_jb.e, out_kb.s, out_kb.e, vlen);
          found = true;
          break;
        }
      }
      if (!found) {
        std::stringstream msg;
        msg << "### ERROR: Unable to find variable " << vWriteName << std::endl;
        PARTHENON_FAIL(msg);
      }
    }
    // write dataset to file
    WRITEH5SLAB2(vWriteName.c_str(), tmpData.data(), file, local_start, local_count,
                 vLocalSpace, vGlobalSpace, property_list);
    if (vlen > 1) {
      H5Sclose(vLocalSpace);
      H5Sclose(vGlobalSpace);
    }
  }
  // close persistent data spaces
  H5Sclose(local_DSpace);
  H5Sclose(global_DSpace);

  // Sriram's hack for faces writes scalar face variables
  // Vector face variables will be written as _n
  // this is a stupidly complicated multi-pass through the variable
  // list, but again will revisit when the time comes to redo
  hsize_t maxVF = 1;
  for (auto &v : ciX.varsFace) {
    const size_t vlen = v->Get(1).GetDim(4);
    maxVF = (maxVF < vlen ? vlen : maxVF);
    //    std::cout << "FOUND FACE: " << v->label() << std::endl;
  }
  std::vector<Real> tmpDataF(
      pm->ndim * (nx1 + 1) * (nx2 + 1) * (nx3 + 1) * maxVF * num_blocks_local, 0);
  Real *dataF = tmpDataF.data();
  for (auto &vwrite : ciX.varsFace) { // for each Face variable we write
    const std::string vWriteName = vwrite->label();
    const hsize_t vlen = vwrite->Get(1).GetDim(4);
    local_count[4] = global_count[4] = vlen;

    local_count[3] += 1;
    global_count[3] += 1;
    auto vLocalSpaceX = H5Screate_simple(5, local_count, NULL);
    auto vGlobalSpaceX = H5Screate_simple(5, global_count, NULL);
    local_count[3] -= 1;
    global_count[3] -= 1;

    local_count[2] += 1;
    global_count[2] += 1;
    auto vLocalSpaceY = H5Screate_simple(5, local_count, NULL);
    auto vGlobalSpaceY = H5Screate_simple(5, global_count, NULL);
    local_count[2] -= 1;
    global_count[2] -= 1;

    local_count[1] += 1;
    global_count[1] += 1;
    auto vLocalSpaceZ = H5Screate_simple(5, local_count, NULL);
    auto vGlobalSpaceZ = H5Screate_simple(5, global_count, NULL);
    local_count[1] -= 1;
    global_count[1] -= 1;

    hsize_t offset = (nx1 + 1) * (nx2 + 1) * (nx3 + 1);
    hsize_t index1 = 0;
    hsize_t index2 = offset;
    hsize_t index3 = 2 * offset;
    for (auto &pmb : pm->block_list) { // for every block1
      auto ci =
          MeshBlockDataIterator<Real>(pmb->meshblock_data.Get(), output_params.variables);
      for (auto &v : ci.varsFace) {
        std::string name = v->label();
        if (name.compare(vWriteName) == 0) {
          // copy data to host

          // Load x direction
          auto v_x = v->Get(1).GetHostMirrorAndCopy();
          LOADVARIABLEONE(index1, dataF, v_x, out_ib.s, out_ib.e + 1, out_jb.s, out_jb.e,
                          out_kb.s, out_kb.e, vlen);
          if (pm->ndim > 1) {
            // Load y direction
            auto v_y = v->Get(2).GetHostMirrorAndCopy();
            LOADVARIABLEONE(index2, dataF, v_y, out_ib.s, out_ib.e, out_jb.s,
                            out_jb.e + 1, out_kb.s, out_kb.e, vlen);
          }
          if (pm->ndim > 2) {
            // Load z direction
            auto v_z = v->Get(3).GetHostMirrorAndCopy();
            LOADVARIABLEONE(index3, dataF, v_z, out_ib.s, out_ib.e, out_jb.s, out_jb.e,
                            out_kb.s, out_kb.e + 1, vlen);
          }
          break;
        }
      }
    }
    // write datasets to file
    // X direction
    {
      std::string nameX = vWriteName + "_x1";
      local_count[3] += 1;
      WRITEH5SLAB2(nameX.c_str(), dataF, file, local_start, local_count, vLocalSpaceX,
                   vGlobalSpaceX, property_list);
      local_count[3] -= 1;
    }
    if (pm->ndim > 1) {
      dataF += offset;
      std::string nameX = vWriteName + "_x2";
      local_count[2] += 1;
      WRITEH5SLAB2(nameX.c_str(), dataF, file, local_start, local_count, vLocalSpaceY,
                   vGlobalSpaceY, property_list);
      local_count[2] -= 1;
    }
    if (pm->ndim > 2) {
      dataF += offset;
      std::string nameX = vWriteName + "_x3";
      local_count[1] += 1;
      WRITEH5SLAB2(nameX.c_str(), dataF, file, local_start, local_count, vLocalSpaceZ,
                   vGlobalSpaceZ, property_list);
      local_count[1] -= 1;
    }
    // close data spaces
    H5Sclose(vLocalSpaceX);
    H5Sclose(vGlobalSpaceX);
    H5Sclose(vLocalSpaceY);
    H5Sclose(vGlobalSpaceY);
    H5Sclose(vLocalSpaceZ);
    H5Sclose(vGlobalSpaceZ);
  }

#ifdef MPI_PARALLEL
  /* release the file access template */
  ierr = H5Pclose(acc_file);
  ierr = MPI_Info_free(&FILE_INFO_TEMPLATE);
#endif

  H5Pclose(property_list);
  H5Fclose(file);

  // advance output parameters
  output_params.file_number++;
  output_params.next_time += output_params.dt;
  pin->SetInteger(output_params.block_name, "file_number", output_params.file_number);
  pin->SetReal(output_params.block_name, "next_time", output_params.next_time);
  return;
#endif
}
void instantiateReader_(RestartReader &rr) {
  // aroutine to instantiate templated routines so they can be used elsewhere
  size_t count;
  IndexRange myBlocks{0, 0};
  auto dataI32 = rr.ReadDataset<int32_t>("xxx", &count);
  auto dataI64 = rr.ReadDataset<int64_t>("xxx", &count);
  auto dataFloat = rr.ReadDataset<float>("xxx", &count);
  auto dataDouble = rr.ReadDataset<double>("xxx", &count);
  std::cout << "dummy routine" << count;
}

} // namespace parthenon
