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
#include <mpi.h>
#include <string>
#include <utility>

#include "H5Tpublic.h"
#include "H5public.h"
#include "globals.hpp"
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
  fh_ = H5F::FromHIDCheck(H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT));

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
  hid_t theHdfType = H5T_C_S1;

  H5D const dset = H5D::FromHIDCheck(H5Dopen2(fh_, dataset, H5P_DEFAULT));
  H5A const attr = H5A::FromHIDCheck(H5Aopen(dset, name, H5P_DEFAULT));
  H5S const dataspace = H5S::FromHIDCheck(H5Aget_space(attr));

  // Allocate array of correct size
  H5T const filetype = H5T::FromHIDCheck(H5Aget_type(attr));
  hsize_t isize = H5Tget_size(filetype);
  isize++;
  if (count != nullptr) {
    *count = isize;
  }

  std::vector<char> s(isize + 1, '\0');
  // Read data from file
  //  H5Aread(attr, theHdfType, static_cast<void *>(s));
  H5T const memType = H5T::FromHIDCheck(H5Tcopy(H5T_C_S1));
  PARTHENON_HDF5_CHECK(H5Tset_size(memType, isize));
  PARTHENON_HDF5_CHECK(H5Aread(attr, memType, s.data()));

  return std::string(s.data());
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

  int rootLevel = pm->GetRootLevel();
  int max_level = pm->GetCurrentLevel() - rootLevel;
  auto nblist = pm->GetNbList();

  const IndexDomain theDomain =
      (output_params.include_ghost_zones ? IndexDomain::entire : IndexDomain::interior);

  // all blocks have the same logical size, so get bounds from first mesh block
  auto &first_mb = *(pm->block_list.front());

  const IndexRange out_ib = first_mb.cellbounds.GetBoundsI(theDomain);
  const IndexRange out_jb = first_mb.cellbounds.GetBoundsJ(theDomain);
  const IndexRange out_kb = first_mb.cellbounds.GetBoundsK(theDomain);

  auto const nx1 = out_ib.e - out_ib.s + 1; // SS mb.block_size.nx1;
  auto const nx2 = out_jb.e - out_jb.s + 1; // SS mb.block_size.nx2;
  auto const nx3 = out_kb.e - out_kb.s + 1; // SS mb.block_size.nx3;

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

  {
    hid_t acc_file = H5P_DEFAULT;

#ifdef MPI_PARALLEL
    /* set the file access template for parallel IO access */
    H5P const acc_parallel_file = H5P::FromHIDCheck(H5Pcreate(H5P_FILE_ACCESS));
    acc_file = acc_parallel_file;

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
#endif

    // now open the file
    H5F const file = H5F::FromHIDCheck(
        H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, acc_file));

    // attributes written here:
    // All ranks write attributes

    // write timestep relevant attributes
    hsize_t nLen;

    { // write input key-value pairs
      std::ostringstream oss;
      pin->ParameterDump(oss);

      // Mesh information
      H5S const localDSpace = H5S::FromHIDCheck(H5Screate(H5S_SCALAR));
      H5D const myDSet = H5D::FromHIDCheck(H5Dcreate(
          file, "/Input", PREDINT32, localDSpace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));

      writeH5ASTRING("File", oss.str(), localDSpace, myDSet);
    }

    {
      H5S const localDSpace = H5S::FromHIDCheck(H5Screate(H5S_SCALAR));
      H5D const myDSet = H5D::FromHIDCheck(H5Dcreate(
          file, "/Info", PREDINT32, localDSpace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));

      if (tm != nullptr) {
        writeH5AI32("NCycle", &(tm->ncycle), localDSpace, myDSet);
        writeH5AF64("Time", &(tm->time), localDSpace, myDSet);
        writeH5AF64("dt", &(tm->dt), localDSpace, myDSet);
      }
      writeH5ASTRING("Coordinates", std::string(first_mb.coords.Name()), localDSpace,
                     myDSet);

      writeH5AI32("NumDims", &pm->ndim, localDSpace, myDSet);

      hsize_t nPE = Globals::nranks;
      writeH5AI32("BlocksPerPE", nblist.data(),
                  H5S::FromHIDCheck(H5Screate_simple(1, &nPE, NULL)), myDSet);
    }

    // Mesh information
    {
      H5S const localDSpace = H5S::FromHIDCheck(H5Screate(H5S_SCALAR));
      H5D const myDSet = H5D::FromHIDCheck(H5Dcreate(
          file, "/Mesh", PREDINT32, localDSpace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));

      {
        int bsize[3] = {first_mb.block_size.nx1, first_mb.block_size.nx2,
                        first_mb.block_size.nx3};
        nLen = 3;
        H5S const localnDSpace = H5S::FromHIDCheck(H5Screate_simple(1, &nLen, NULL));
        writeH5AI32("blockSize", bsize, localnDSpace, myDSet);
        int iGhost = (output_params.include_ghost_zones ? 1 : 0);
        writeH5AI32("includesGhost", &iGhost, localDSpace, myDSet);
      }

      writeH5AI32("nbtotal", &pm->nbtotal, localDSpace, myDSet);
      writeH5AI32("nbnew", &pm->nbnew, localDSpace, myDSet);
      writeH5AI32("nbdel", &pm->nbdel, localDSpace, myDSet);
      writeH5AI32("rootLevel", &rootLevel, localDSpace, myDSet);
      writeH5AI32("MaxLevel", &max_level, localDSpace, myDSet);

      { // refinement flag
        int refine = (pm->adaptive ? 1 : 0);
        writeH5AI32("refine", &refine, localDSpace, myDSet);

        int multilevel = (pm->multilevel ? 1 : 0);
        writeH5AI32("multilevel", &multilevel, localDSpace, myDSet);
      }

      { // mesh bounds
        const auto &rs = pm->mesh_size;
        const Real limits[6] = {rs.x1min, rs.x2min, rs.x3min,
                                rs.x1max, rs.x2max, rs.x3max};
        const Real ratios[3] = {rs.x1rat, rs.x2rat, rs.x3rat};
        nLen = 6;
        writeH5AF64("bounds", limits, H5S::FromHIDCheck(H5Screate_simple(1, &nLen, NULL)),
                    myDSet);

        nLen = 3;
        writeH5AF64("ratios", ratios, H5S::FromHIDCheck(H5Screate_simple(1, &nLen, NULL)),
                    myDSet);
      }

      { // boundary conditions
        nLen = 6;
        int bcsi[6];
        for (int ib = 0; ib < 6; ib++) {
          bcsi[ib] = static_cast<int>(pm->mesh_bcs[ib]);
        }
        writeH5AI32("bc", bcsi, H5S::FromHIDCheck(H5Screate_simple(1, &nLen, NULL)),
                    myDSet);
      }
    }

    // end mesh section

    // write blocks
    // MeshBlock information
    // Write mesh coordinates to file
    hsize_t local_start[5], global_count[5], local_count[5];

    local_start[0] = 0;
    local_start[1] = 0;
    local_start[2] = 0;
    local_start[3] = 0;
    local_start[4] = 0;
    for (int i = 0; i < Globals::my_rank; i++) {
      local_start[0] += nblist[i];
    }
    H5P const property_list = H5P::FromHIDCheck(H5Pcreate(H5P_DATASET_XFER));
#ifdef MPI_PARALLEL
    PARTHENON_HDF5_CHECK(H5Pset_dxpl_mpio(property_list, H5FD_MPIO_COLLECTIVE));
#endif

    // set starting point in hyperslab for our blocks and
    // number of blocks on our PE

    // open blocks tab
    H5G const gBlocks = H5G::FromHIDCheck(
        H5Gcreate(file, "/Blocks", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));

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
      WRITEH5SLABI32("loc.level-gid-lid-cnghost-gflag", tmpID.data(), gBlocks,
                     local_start, local_count, global_count, property_list);
    }

    // write variables
    {

      // first we need to get list of variables, because sparse variables are only
      // expanded on some blocks, we need to look at the list of variables on each block
      struct VarInfo {
        VarInfo(const std::string label, int vlen) : label(label), vlen(vlen) {
          if (vlen == 0) {
            std::stringstream msg;
            msg << "### ERROR: Got variable " << label << " with length 0" << std::endl;
            PARTHENON_FAIL(msg);
          }
        }

        explicit VarInfo(const std::shared_ptr<CellVariable<Real>> &var)
            : VarInfo(var->label(), var->IsSparse() ? -var->GetDim(4) : var->GetDim(4)) {}

        std::string label;

        // we also encode whether the variable is sparse or not in this field, positive:
        // not sparse, negative: sparse
        int vlen;

        // so we can put VarInfo into a set
        bool operator<(const VarInfo &other) const {
          if ((label == other.label) && (vlen != other.vlen)) {
            // variables with the same label must have the same lengths
            std::stringstream msg;
            msg << "### ERROR: Got variable " << label
                << " with multiple different lengths" << std::endl;
            PARTHENON_FAIL(msg);
          }

          return label < other.label;
        }
      };

      auto metadata_filter = {parthenon::Metadata::Independent,
                              parthenon::Metadata::Restart};
      std::set<VarInfo> all_unique_vars;
      for (auto &pmb : pm->block_list) {
        auto ci =
            MeshBlockDataIterator<Real>(pmb->meshblock_data.Get(), metadata_filter, true);
        for (auto &v : ci.vars) {
          VarInfo vinfo(v);
          all_unique_vars.insert(vinfo);
        }
      }

#ifdef MPI_PARALLEL
      {
        // we need to do a global allgather to get the global list of unique variables to
        // be written to the HDF5 file

        // the label buffer contains all labels of the unique variables on this rank
        // separated by \t, e.g.: "label0\tlabel1\tlabel2\t"
        std::string label_buffer;
        std::vector<int> vlen_buffer(all_unique_vars.size(), 0);

        size_t idx = 0;
        for (const auto &vi : all_unique_vars) {
          label_buffer += vi.label + "\t";
          vlen_buffer[idx++] = vi.vlen;
        }

        // first we need to communicate the lengths of the label_buffer and vlen_buffer to
        // all ranks, 2 ints per rank: first int: label_buffer length, second int:
        // vlen_buffer length
        std::vector<int> buffer_lengths(2 * Globals::nranks, 0);
        buffer_lengths[Globals::my_rank * 2 + 0] = int(label_buffer.size());
        buffer_lengths[Globals::my_rank * 2 + 1] = int(vlen_buffer.size());

        PARTHENON_MPI_CHECK(MPI_Allgather(MPI_IN_PLACE, 2, MPI_INT, buffer_lengths.data(),
                                          2, MPI_INT, MPI_COMM_WORLD));

        // now do an Allgatherv combining label_buffer and vlen_buffer from all ranks
        std::vector<int> label_lengths(Globals::nranks, 0);
        std::vector<int> label_offsets(Globals::nranks, 0);
        std::vector<int> vlen_lengths(Globals::nranks, 0);
        std::vector<int> vlen_offsets(Globals::nranks, 0);

        int label_offset = 0;
        int vlen_offset = 0;
        for (int n = 0; n < Globals::nranks; ++n) {
          label_offsets[n] = label_offset;
          vlen_offsets[n] = vlen_offset;

          label_lengths[n] = buffer_lengths[n * 2 + 0];
          vlen_lengths[n] = buffer_lengths[n * 2 + 1];

          label_offset += label_lengths[n];
          vlen_offset += vlen_lengths[n];
        }

        // result buffers with global data
        std::vector<char> all_labels_buffer(label_offset, '\0');
        std::vector<int> all_vlen(vlen_offset, 0);

        // fill in our values in global buffers
        memcpy(all_labels_buffer.data() + label_offsets[Globals::my_rank],
               label_buffer.data(), label_buffer.size() * sizeof(char));
        memcpy(all_vlen.data() + vlen_offsets[Globals::my_rank], vlen_buffer.data(),
               vlen_buffer.size() * sizeof(int));

        PARTHENON_MPI_CHECK(MPI_Allgatherv(MPI_IN_PLACE, label_lengths[Globals::my_rank],
                                           MPI_BYTE, all_labels_buffer.data(),
                                           label_lengths.data(), label_offsets.data(),
                                           MPI_BYTE, MPI_COMM_WORLD));

        PARTHENON_MPI_CHECK(MPI_Allgatherv(MPI_IN_PLACE, vlen_lengths[Globals::my_rank],
                                           MPI_INT, all_vlen.data(), vlen_lengths.data(),
                                           vlen_offsets.data(), MPI_INT, MPI_COMM_WORLD));

        // unpack labels
        std::vector<std::string> all_labels;
        const char *curr = all_labels_buffer.data();
        const char *const end = curr + all_labels_buffer.size();

        while (curr < end) {
          const auto tab = strchr(curr, '\t');
          if (tab == nullptr) {
            std::stringstream msg;
            msg << "### ERROR: all_labels_buffer does not end with \\t" << std::endl;
            PARTHENON_FAIL(msg);
          }

          if (tab == curr) {
            std::stringstream msg;
            msg << "### ERROR: Got an empty label" << std::endl;
            PARTHENON_FAIL(msg);
          }

          std::string label(curr, tab - curr);
          all_labels.push_back(label);
          curr = tab + 1;
        }

        if (all_labels.size() != all_vlen.size()) {
          printf("all_labels: %lu\n", all_labels.size());
          for (size_t i = 0; i < all_labels.size(); ++i)
            printf("%4lu: %s\n", i, all_labels[i].c_str());
          printf("all_vlen: %lu\n", all_vlen.size());
          for (size_t i = 0; i < all_vlen.size(); ++i)
            printf("%4lu: %i\n", i, all_vlen[i]);

          std::stringstream msg;
          msg << "### ERROR: all_labels and all_vlen have different sizes" << std::endl;
          PARTHENON_FAIL(msg);
        }

        // finally make list of all unique variables
        for (size_t i = 0; i < all_labels.size(); ++i) {
          VarInfo vinfo(all_labels[i], all_vlen[i]);
          all_unique_vars.insert(vinfo);
        }
      }
#endif

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

      H5S const local_DSpace = H5S::FromHIDCheck(H5Screate_simple(5, local_count, NULL));
      H5S const global_DSpace =
          H5S::FromHIDCheck(H5Screate_simple(5, global_count, NULL));

      // while we could do this as n variables and load all variables for
      // a block at one time, this is memory-expensive.  I think it is
      // well worth the multiple iterations through the blocks to load up
      // one variable at a time.  Besides most of the time will be spent
      // writing the HDF5 file to disk anyway...
      // If I'm wrong about this, we can always rewrite this later.
      // Sriram

      // We need to add information about the sparse variables to the HDF5 file, namely:
      // 1) Which variables are sparse
      // 2) Is a sparse id of a particular sparse variable expanded on a given block
      //
      // This information is stored in the dataset called "SparseInfo". The data set
      // contains an attribute "SparseFields" that is a vector of strings with the names
      // of the sparse fields (field name with sparse id, i.e. "bar_28", "bar_7", foo_1",
      // "foo_145"). The field names are in alphabetical order, which is the same order
      // they show up in all_unique_vars (because it's a sorted set).
      //
      // The dataset SparseInfo itself is a 2D array of bools. The first index is the
      // global block index and the second index is the sparse field (same order as the
      // SparseFields attribute). SparseInfo[b][v] is true if the sparse field with index
      // v is expanded on the block with index b, otherwise the value is false

      std::vector<std::string> sparse_names;
      std::unordered_map<std::string, size_t> sparse_field_idx;
      for (auto &vinfo : all_unique_vars) {
        if (vinfo.vlen < 0) {
          sparse_field_idx.insert({vinfo.label, sparse_names.size()});
          sparse_names.push_back(vinfo.label);
        }
      }

      hsize_t num_sparse = sparse_names.size();
      // can't use std::vector here because std::vector<hbool_t> is the same as
      // std::vector<bool> and it doesn't have .data() member
      std::unique_ptr<hbool_t[]> sparse_expanded(
          new hbool_t[num_blocks_local * num_sparse]);

      const hsize_t varSize = nx3 * nx2 * nx1;
      for (auto &vinfo : all_unique_vars) { // for each variable we write
        const std::string vWriteName = vinfo.label;
        hid_t vLocalSpace, vGlobalSpace;
        H5S vLocalSpaceNew, vGlobalSpaceNew;
        auto &mb = *(pm->block_list.front());
        const bool is_sparse = (vinfo.vlen < 0);
        const hsize_t vlen = abs(vinfo.vlen);
        local_count[4] = global_count[4] = vlen;
        std::vector<Real> tmpData(varSize * vlen * num_blocks_local);

        // create spaces if required
        if (vlen == 1) {
          vLocalSpace = local_DSpace;
          vGlobalSpace = global_DSpace;
        } else {
          vLocalSpace = vLocalSpaceNew =
              H5S::FromHIDCheck(H5Screate_simple(5, local_count, NULL));
          vGlobalSpace = vGlobalSpaceNew =
              H5S::FromHIDCheck(H5Screate_simple(5, global_count, NULL));
        }

        // load up data
        hsize_t index = 0;
        bool found_any = false;
        for (size_t b_idx = 0; b_idx < num_blocks_local; ++b_idx) {
          const auto &pmb = pm->block_list[b_idx];
          bool found = false;
          auto ci = MeshBlockDataIterator<Real>(pmb->meshblock_data.Get(),
                                                metadata_filter, true);
          for (auto &v : ci.vars) {
            // Note index 4 transposed to interior
            if (vWriteName.compare(v->label()) == 0) {
              auto v_h = v->data.GetHostMirrorAndCopy();
              LOADVARIABLEONE(index, tmpData.data(), v_h, out_ib.s, out_ib.e, out_jb.s,
                              out_jb.e, out_kb.s, out_kb.e, vlen);
              found = true;
              break;
            }
          }

          if (is_sparse) {
            size_t sparse_idx = sparse_field_idx.at(vinfo.label);
            sparse_expanded[b_idx * num_sparse + sparse_idx] = found;
          }

          if (!found) {
            if (is_sparse) {
              hsize_t N = varSize * vlen;
              memset(tmpData.data() + index, 0, N * sizeof(Real));
              index += N;
            } else {
              std::stringstream msg;
              msg << "### ERROR: Unable to find dense variable " << vWriteName
                  << std::endl;
              PARTHENON_FAIL(msg);
            }
          } else {
            found_any = true;
          }
        }

        if (found_any) {
          // write dataset to file
          WRITEH5SLAB2(vWriteName.c_str(), tmpData.data(), file, local_start, local_count,
                       vLocalSpace, vGlobalSpace, property_list);
        }
      }

      // write SparseInfo
      local_count[0] = num_blocks_local;
      global_count[0] = max_blocks_global;
      local_count[1] = global_count[1] = num_sparse;

      WRITEH5SLABBOOL("SparseInfo", sparse_expanded.get(), file, local_start, local_count,
                      global_count, property_list);

      // write names of sparse fields as attribute
      {
        // make vector of const char*
        std::vector<const char *> names(num_sparse);
        for (size_t i = 0; i < num_sparse; ++i)
          names[i] = sparse_names[i].c_str();

        const H5S attr_space =
            H5S::FromHIDCheck(H5Screate_simple(1, &num_sparse, &num_sparse));
        const H5T atype = H5T::FromHIDCheck(H5Tcopy(H5T_C_S1));
        PARTHENON_HDF5_CHECK(H5Tset_size(atype, H5T_VARIABLE));

        const H5D dset = H5D::FromHIDCheck(H5Dopen2(file, "SparseInfo", H5P_DEFAULT));

        const H5A attribute = H5A::FromHIDCheck(
            H5Acreate(dset, "SparseFields", atype, attr_space, H5P_DEFAULT, H5P_DEFAULT));
        PARTHENON_HDF5_CHECK(H5Awrite(attribute, atype, names.data()));
      }
    }
  }

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
  auto dataI32 = rr.ReadDataset<int32_t>("xxx", &count);
  auto dataI64 = rr.ReadDataset<int64_t>("xxx", &count);
  auto dataFloat = rr.ReadDataset<float>("xxx", &count);
  auto dataDouble = rr.ReadDataset<double>("xxx", &count);
  std::cout << "dummy routine" << count;
}

} // namespace parthenon
