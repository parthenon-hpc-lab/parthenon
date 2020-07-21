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

#include "outputs/outputs.hpp"
#include "outputs/parthenon_hdf5.hpp"

namespace parthenon {

//----------------------------------------------------------------------------------------
//! \fn void RestartOutput::WriteOutputFile(Mesh *pm, ParameterInput *pin, bool flag)
//  \brief Cycles over all MeshBlocks and writes data to a single restart file.
void RestartOutput::WriteOutputFile(Mesh *pm, ParameterInput *pin, SimTime *tm) {
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
  MeshBlock *pmb = pm->pblock;
  hsize_t max_blocks_global = pm->nbtotal;
  hsize_t num_blocks_local = 0;

  const IndexDomain interior = IndexDomain::interior;

  // shooting a blank just for getting the variable names
  const IndexRange out_ib = pmb->cellbounds.GetBoundsI(interior);
  const IndexRange out_jb = pmb->cellbounds.GetBoundsJ(interior);
  const IndexRange out_kb = pmb->cellbounds.GetBoundsK(interior);

  while (pmb != nullptr) {
    num_blocks_local++;
    pmb = pmb->next;
  }
  pmb = pm->pblock;
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
  }
  status = writeH5ASTRING("Coordinates", std::string(pmb->coords.Name()), file,
                          localDSpace, myDSet);

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

  auto &nx1 = pmb->block_size.nx1;
  auto &nx2 = pmb->block_size.nx2;
  auto &nx3 = pmb->block_size.nx3;
  int bsize[3] = {nx1, nx2, nx3};
  nLen = 3;
  localnDSpace = H5Screate_simple(1, &nLen, NULL);
  status = writeH5AI32("blockSize", bsize, file, localnDSpace, myDSet);
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
    status = writeH5AF64("ratios", limits, file, localnDSpace, myDSet);
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
    Real *tmpData = new Real[num_blocks_local * 3];
    local_count[0] = num_blocks_local;
    global_count[0] = max_blocks_global;
    pmb = pm->pblock;
    int i = 0;
    while (pmb != nullptr) {
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
      pmb = pmb->next;
    }
    local_count[1] = global_count[1] = pm->ndim;
    WRITEH5SLABDOUBLE("xmin", tmpData, gBlocks, local_start, local_count, global_count,
                      property_list);
    delete[] tmpData;
  }

  // write Block ID
  {
    // LOC.lx1,2,3
    hsize_t n;
    int i;

    n = 3;
    auto *tmpLoc = new int64_t[num_blocks_local * n];
    local_count[1] = global_count[1] = n;
    local_count[0] = num_blocks_local;
    global_count[0] = max_blocks_global;
    pmb = pm->pblock;
    i = 0;
    while (pmb != nullptr) {
      tmpLoc[i++] = pmb->loc.lx1;
      tmpLoc[i++] = pmb->loc.lx2;
      tmpLoc[i++] = pmb->loc.lx3;
      pmb = pmb->next;
    }
    WRITEH5SLABI64("loc.lx123", tmpLoc, gBlocks, local_start, local_count, global_count,
                   property_list);
    delete[] tmpLoc;

    // (LOC.)level, GID, LID, cnghost, gflag
    n = 5;
    auto *tmpID = new int[num_blocks_local * n];
    local_count[1] = global_count[1] = n;
    local_count[0] = num_blocks_local;
    global_count[0] = max_blocks_global;
    pmb = pm->pblock;
    i = 0;
    while (pmb != nullptr) {
      tmpID[i++] = pmb->loc.level;
      tmpID[i++] = pmb->gid;
      tmpID[i++] = pmb->lid;
      tmpID[i++] = pmb->cnghost;
      tmpID[i++] = pmb->gflag;
      pmb = pmb->next;
    }
    WRITEH5SLABI32("loc.level-gid-lid-cnghost-gflag", tmpID, gBlocks, local_start,
                   local_count, global_count, property_list);
    delete[] tmpID;
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

  // Get an iterator on block 0 for variable listing
  auto ciX = ContainerIterator<Real>(pm->pblock->real_containers.Get(),
                                     {parthenon::Metadata::Restart});
  for (auto &vwrite : ciX.vars) { // for each variable we write
    const std::string vWriteName = vwrite->label();
    hid_t vLocalSpace, vGlobalSpace;
    pmb = pm->pblock;
    const hsize_t vlen = vwrite->GetDim(4);
    local_count[4] = global_count[4] = vlen;
    Real *tmpData = new Real[varSize * vlen * num_blocks_local];

    // create spaces if required
    if (vlen == 1) {
      vLocalSpace = local_DSpace;
      vGlobalSpace = global_DSpace;
    } else {
      vLocalSpace = H5Screate_simple(5, local_count, NULL);
      vGlobalSpace = H5Screate_simple(5, global_count, NULL);
    }

    // load up data
    while (pmb != nullptr) { // for every block
      auto ci = ContainerIterator<Real>(pmb->real_containers.Get(),
                                        {parthenon::Metadata::Restart});
      for (auto &v : ci.vars) {
        std::string name = v->label();
        if (name.compare(vWriteName) != 0) {
          // skip, not interested in this variable
          continue;
        }
        auto v_h = (*v).data.GetHostMirrorAndCopy();
        hsize_t index = pmb->lid * varSize * vlen;
        // Note index 4 transposed to interior
        for (int k = out_kb.s; k <= out_kb.e; k++) {
          for (int j = out_jb.s; j <= out_jb.e; j++) {
            for (int i = out_ib.s; i <= out_ib.e; i++) {
              for (int l = 0; l < vlen; l++, index++) {
                tmpData[index] = v_h(l, k, j, i);
              }
            }
          }
        }
      }
      pmb = pmb->next;
    }
    // write dataset to file
    WRITEH5SLAB2(vWriteName.c_str(), tmpData, file, local_start, local_count,
                 vLocalSpace, vGlobalSpace, property_list);
    //    WRITEH5SLAB(vWriteName.c_str(), tmpData, file, local_start, local_count,
    //    vLocalSpace,
    //           vGlobalSpace, property_list);
    if (vlen > 1) {
      H5Sclose(vLocalSpace);
      H5Sclose(vGlobalSpace);
    }
    delete[] tmpData;
  }
  // close persistent data spaces
  H5Sclose(local_DSpace);
  H5Sclose(global_DSpace);

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
} // namespace parthenon
