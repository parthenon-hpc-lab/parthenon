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

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>

#include "parthenon_mpi.hpp"

#include "athena.hpp"
#include "coordinates/coordinates.hpp"
#include "globals.hpp"
#include "interface/container_iterator.hpp"
#include "mesh/mesh.hpp"
#include "outputs/outputs.hpp"
#include "parameter_input.hpp"
#include "parthenon_arrays.hpp"

// Only proceed if HDF5 output enabled
#ifdef HDF5OUTPUT

#include <hdf5.h>

#define PREDINT32 H5T_NATIVE_INT32
#define PREDFLOAT64 H5T_NATIVE_DOUBLE
#define PREDCHAR H5T_NATIVE_CHAR

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
                                     const std::string &dims321) {
  // writes a slab reference to file

  const std::string prefix = "      ";
  fid << prefix << R"(<Attribute Name=")" << name << R"(" Center="Cell")";
  if (vlen > 1) {
    fid << R"( AttributeType="Vector")"
        << R"( Dimensions=")" << dims321 << " " << vlen << R"(")";
  }
  fid << ">" << std::endl;
  fid << prefix << "  "
      << R"(<DataItem ItemType="HyperSlab" Dimensions=")" << dims321 << " " << vlen
      << R"(">)" << std::endl;
  fid << prefix << "    "
      << R"(<DataItem Dimensions="3 5" NumberType="Int" Format="XML">)" << iblock
      << " 0 0 0 0 1 1 1 1 1 1 " << dims321 << " " << vlen << "</DataItem>" << std::endl;

  writeXdmfArrayRef(fid, prefix + "    ", hdfFile + ":/", name, dims, ndims, "Float", 8);
  fid << prefix << "  "
      << "</DataItem>" << std::endl;
  fid << prefix << "</Attribute>" << std::endl;
  return;
}

static herr_t writeH5AI32(const char *name, const int *pData, hid_t &file,
                          const hid_t &dSpace, const hid_t &dSet) {
  // write an attribute to file
  herr_t status; // assumption that multiple errors are stacked in calls.
  hid_t attribute;
  attribute = H5Acreate(dSet, name, PREDINT32, dSpace, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Awrite(attribute, PREDINT32, pData);
  status = H5Aclose(attribute);
  return status;
}

static herr_t writeH5AF64(const char *name, const Real *pData, hid_t &file,
                          const hid_t &dSpace, const hid_t &dSet) {
  // write an attribute to file
  herr_t status; // assumption that multiple errors are stacked in calls.
  hid_t attribute;
  attribute = H5Acreate(dSet, name, PREDFLOAT64, dSpace, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Awrite(attribute, PREDFLOAT64, pData);
  status = H5Aclose(attribute);
  return status;
}

static herr_t writeH5ASTRING(const char *name, const std::string pData, hid_t &file,
                             const hid_t &dSpace, const hid_t &dSet) {
  auto atype = H5Tcopy(H5T_C_S1);
  auto status = H5Tset_size(atype, pData.length());
  status = H5Tset_strpad(atype, H5T_STR_NULLTERM);
  auto attribute = H5Acreate(dSet, name, atype, dSpace, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Awrite(attribute, atype, pData.c_str());
  status = H5Aclose(attribute);
  return status;
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
  MeshBlock *pmb;
  hsize_t dims[5] = {0, 0, 0, 0, 0};

  pmb = pm->pblock;

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
  pmb = pm->pblock;
  dims[0] = pm->nbtotal;
  std::string dims321 =
      std::to_string(nx3) + " " + std::to_string(nx2) + " " + std::to_string(nx1);

  int ndims = 5;

  // same set of variables for all grids so use only one container
  auto ciX = ContainerIterator<Real>(pmb->real_containers.Get(), output_params.variables);
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
      writeXdmfSlabVariableRef(xdmf, name, hdfFile, ib, vlen, ndims, dims, dims321);
    }
    xdmf << "      </Grid>" << std::endl;
  }
  xdmf << "    </Grid>" << std::endl;
  xdmf << "  </Domain>" << std::endl;
  xdmf << "</Xdmf>" << std::endl;
  xdmf.close();

  return;
}

// loads a variable
#define LOADVARIABLE(dst, pmb, var, out_is, out_ie, out_js, out_je, out_ks, out_ke)      \
  {                                                                                      \
    int index = 0;                                                                       \
    while (pmb != nullptr) {                                                             \
      for (int k = out_ks; k <= out_ke; k++) {                                           \
        for (int j = out_js; j <= out_je; j++) {                                         \
          for (int i = out_is; i <= out_ie; i++) {                                       \
            tmpData[index] = var(k, j, i);                                               \
            index++;                                                                     \
          }                                                                              \
        }                                                                                \
      }                                                                                  \
      pmb = pmb->next;                                                                   \
    }                                                                                    \
  }

#define WRITEH5SLAB2(name, pData, theLocation, Starts, Counts, lDSpace, gDSpace, plist)  \
  {                                                                                      \
    hid_t gDSet = H5Dcreate(theLocation, name, H5T_NATIVE_DOUBLE, gDSpace, H5P_DEFAULT,  \
                            H5P_DEFAULT, H5P_DEFAULT);                                   \
    H5Sselect_hyperslab(gDSpace, H5S_SELECT_SET, Starts, NULL, Counts, NULL);            \
    H5Dwrite(gDSet, H5T_NATIVE_DOUBLE, lDSpace, gDSpace, plist, pData);                  \
    H5Dclose(gDSet);                                                                     \
  }
#define WRITEH5SLAB(name, pData, theLocation, localStart, localCount, globalCount,       \
                    plist)                                                               \
  {                                                                                      \
    hid_t lDSpace = H5Screate_simple(2, localCount, NULL);                               \
    hid_t gDSpace = H5Screate_simple(2, globalCount, NULL);                              \
    WRITEH5SLAB2(name, pData, theLocation, localStart, localCount, lDSpace, gDSpace,     \
                 plist);                                                                 \
    H5Sclose(gDSpace);                                                                   \
    H5Sclose(lDSpace);                                                                   \
  }

//----------------------------------------------------------------------------------------
//! \fn void PHDF5Output:::WriteOutputFile(Mesh *pm, ParameterInput *pin, bool flag)
//  \brief Cycles over all MeshBlocks and writes OutputData in the Parthenon HDF5 format,
//         one file per output using parallel IO.
void PHDF5Output::WriteOutputFile(Mesh *pm, ParameterInput *pin, SimTime *tm) {
  // writes all graphics variables to hdf file
  // HDF5 structures
  // Also writes companion xdmf file
  MeshBlock *pmb = pm->pblock;
  int max_blocks_global = pm->nbtotal;
  int num_blocks_local = 0;

  // shooting a blank just for getting the variable names
  out_is = pmb->is;
  out_ie = pmb->ie;
  out_js = pmb->js;
  out_je = pmb->je;
  out_ks = pmb->ks;
  out_ke = pmb->ke;
  if (output_params.include_ghost_zones) {
    out_is -= NGHOST;
    out_ie += NGHOST;
    if (out_js != out_je) {
      out_js -= NGHOST;
      out_je += NGHOST;
    }
    if (out_ks != out_ke) {
      out_ks -= NGHOST;
      out_ke += NGHOST;
    }
  }

  while (pmb != nullptr) {
    num_blocks_local++;
    pmb = pmb->next;
  }
  pmb = pm->pblock;
  // set output size
  nx1 = pmb->block_size.nx1;
  nx2 = pmb->block_size.nx2;
  nx3 = pmb->block_size.nx3;
  if (output_params.include_ghost_zones) {
    nx1 += 2 * NGHOST;
    if (nx2 > 1) nx2 += 2 * NGHOST;
    if (nx3 > 1) nx3 += 2 * NGHOST;
  }

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

  // write timestep relevant attributes
  hid_t localDSpace, myDSet;
  herr_t status;

  // attributes written here:
  // All ranks write attributes
  localDSpace = H5Screate(H5S_SCALAR);
  myDSet = H5Dcreate(file, "/Info", PREDINT32, localDSpace, H5P_DEFAULT, H5P_DEFAULT,
                     H5P_DEFAULT);

  int max_level = pm->GetCurrentLevel() - pm->GetRootLevel();
  if (tm != nullptr) {
    status = writeH5AI32("NCycle", &(tm->ncycle), file, localDSpace, myDSet);
    status = writeH5AF64("Time", &(tm->time), file, localDSpace, myDSet);
  }
  status = writeH5AI32("NumDims", &pm->ndim, file, localDSpace, myDSet);
  status = writeH5AI32("NumMeshBlocks", &pm->nbtotal, file, localDSpace, myDSet);
  status = writeH5AI32("MaxLevel", &max_level, file, localDSpace, myDSet);
  // write whether we include ghost cells or not
  int iTmp = (output_params.include_ghost_zones ? 1 : 0);
  status = writeH5AI32("IncludesGhost", &iTmp, file, localDSpace, myDSet);
  // write number of ghost cells in simulation
  iTmp = NGHOST;
  status = writeH5AI32("NGhost", &iTmp, file, localDSpace, myDSet);
  status = writeH5ASTRING("Coordinates", std::string(pmb->coords.Name()), file,
                          localDSpace, myDSet);

  // close scalar space
  status = H5Sclose(localDSpace);
  hsize_t nPE = Globals::nranks;
  localDSpace = H5Screate_simple(1, &nPE, NULL);
  auto nblist = pm->GetNbList();
  status = writeH5AI32("BlocksPerPE", nblist.data(), file, localDSpace, myDSet);
  status = H5Sclose(localDSpace);

  // open vector space
  // close data spaces and data set
  // write mesh block size
  int meshblock_size[3] = {nx1, nx2, nx3};
  const hsize_t xDims[1] = {3};
  localDSpace = H5Screate_simple(1, xDims, NULL);
  status = writeH5AI32("MeshBlockSize", meshblock_size, file, localDSpace, myDSet);

  // close space and set
  status = H5Sclose(localDSpace);
  status = H5Dclose(myDSet);

  // allocate space for largest size variable
  auto ciX =
      ContainerIterator<Real>(pm->pblock->real_containers.Get(), output_params.variables);
  size_t maxV = 3;
  hsize_t sumDim4AllVars = 0;
  for (auto &v : ciX.vars) {
    const size_t vlen = v->GetDim(4);
    sumDim4AllVars += vlen;
    maxV = (maxV < vlen ? vlen : maxV);
  }

  Real *tmpData = new Real[(nx1 + 1) * (nx2 + 1) * (nx3 + 1) * maxV * num_blocks_local];
  for (int i = 0; i < (nx1 + 1) * (nx2 + 1) * (nx3 + 1) * maxV * num_blocks_local; i++)
    tmpData[i] = -1.25;

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

  // open locations tab
  gLocations = H5Gcreate(file, "/Locations", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  // write X coordinates
  local_count[0] = num_blocks_local;
  global_count[0] = max_blocks_global;

  pmb = pm->pblock;
  LOADVARIABLE(tmpData, pmb, pmb->coords.x1f, out_is, out_ie + 1, 0, 0, 0, 0);
  local_count[1] = global_count[1] = nx1 + 1;
  WRITEH5SLAB("x", tmpData, gLocations, local_start, local_count, global_count,
              property_list);

  // write Y coordinates
  pmb = pm->pblock;
  LOADVARIABLE(tmpData, pmb, pmb->coords.x2f, 0, 0, out_js, out_je + 1, 0, 0);
  local_count[1] = global_count[1] = nx2 + 1;
  WRITEH5SLAB("y", tmpData, gLocations, local_start, local_count, global_count,
              property_list);

  // write Z coordinates
  pmb = pm->pblock;
  LOADVARIABLE(tmpData, pmb, pmb->coords.x3f, 0, 0, 0, 0, out_ks, out_ke + 1);
  local_count[1] = global_count[1] = nx3 + 1;
  WRITEH5SLAB("z", tmpData, gLocations, local_start, local_count, global_count,
              property_list);

  // close locations tab
  H5Gclose(gLocations);

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

  // this is a stupidly complicated multi-pass through the variable
  // list, but again will revisit when the time comes to redo
  for (auto &vwrite : ciX.vars) { // for each variable we write
    const std::string vWriteName = vwrite->label();
    hid_t vLocalSpace, vGlobalSpace;
    pmb = pm->pblock;
    const hsize_t vlen = vwrite->GetDim(4);
    local_count[4] = global_count[4] = vlen;

    if (vlen == 1) {
      vLocalSpace = local_DSpace;
      vGlobalSpace = global_DSpace;
    } else {
      vLocalSpace = H5Screate_simple(5, local_count, NULL);
      vGlobalSpace = H5Screate_simple(5, global_count, NULL);
    }

    while (pmb != nullptr) { // for every block
      auto ci =
          ContainerIterator<Real>(pmb->real_containers.Get(), output_params.variables);
      for (auto &v : ci.vars) {
        std::string name = v->label();
        if (name.compare(vWriteName) != 0) {
          // skip, not interested in this variable
          continue;
        }
        hsize_t index = pmb->lid * varSize * vlen;
        if (vlen == 1) {
          for (int k = out_ks; k <= out_ke; k++) {
            for (int j = out_js; j <= out_je; j++) {
              for (int i = out_is; i <= out_ie; i++, index++) {
                tmpData[index] = (*v)(k, j, i);
              }
            }
          }
        } else { // shuffle and use new dataspace
          for (int k = out_ks; k <= out_ke; k++) {
            for (int j = out_js; j <= out_je; j++) {
              for (int i = out_is; i <= out_ie; i++) {
                for (int l = 0; l < vlen; l++, index++) {
                  tmpData[index] = (*v)(l, k, j, i);
                }
              }
            }
          }
        }
      }
      pmb = pmb->next;
    }
    // write dataset to file
    WRITEH5SLAB2(vWriteName.c_str(), tmpData, file, local_start, local_count, vLocalSpace,
                 vGlobalSpace, property_list);
    if (vlen > 1) {
      H5Sclose(vLocalSpace);
      H5Sclose(vGlobalSpace);
    }
  }
  // close data spaces
  H5Sclose(local_DSpace);
  H5Sclose(global_DSpace);

#ifdef MPI_PARALLEL
  /* release the file access template */
  ierr = H5Pclose(acc_file);
  ierr = MPI_Info_free(&FILE_INFO_TEMPLATE);
#endif

  H5Pclose(property_list);
  H5Fclose(file);

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
