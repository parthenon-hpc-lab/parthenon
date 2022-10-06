//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2020-2022 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020-2022. Triad National Security, LLC. All rights reserved.
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

// options for building
#include "config.hpp"
#include "globals.hpp"
#include "utils/error_checking.hpp"

// Only proceed if HDF5 output enabled
#ifdef ENABLE_HDF5

#include <algorithm>
#include <memory>
#include <set>
#include <type_traits>
#include <unordered_map>

#include "driver/driver.hpp"
#include "interface/metadata.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "outputs/outputs.hpp"
#include "outputs/parthenon_hdf5.hpp"
#include "utils/string_utils.hpp"

namespace parthenon {
namespace HDF5 {

// template specializations for std::string
template <>
void HDF5WriteAttribute(const std::string &name, const std::vector<std::string> &values,
                        hid_t location) {
  std::vector<const char *> char_ptrs(values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    char_ptrs[i] = values[i].c_str();
  }
  HDF5WriteAttribute(name, char_ptrs, location);
}

template <>
std::vector<std::string> HDF5ReadAttributeVec(hid_t location, const std::string &name) {
  // get strings as char pointers, HDF5 will allocate the memory and we need to free it
  auto char_ptrs = HDF5ReadAttributeVec<char *>(location, name);

  // make strings out of char pointers, which copies the memory and then free the memeory
  std::vector<std::string> res(char_ptrs.size());
  for (size_t i = 0; i < res.size(); ++i) {
    res[i] = std::string(char_ptrs[i]);
    free(char_ptrs[i]);
  }

  return res;
}

// template specialization for bool
template <>
void HDF5WriteAttribute(const std::string &name, const std::vector<bool> &values,
                        hid_t location) {
  // can't use std::vector here because std::vector<bool>  doesn't have .data() member
  std::unique_ptr<hbool_t[]> data(new hbool_t[values.size()]);
  for (size_t i = 0; i < values.size(); ++i) {
    data[i] = values[i];
  }
  HDF5WriteAttribute(name, values.size(), data.get(), location);
}

} // namespace HDF5

using namespace HDF5;

// Helper struct containing some information about a variable
struct VarInfo {
  std::string label;
  int vlen;
  int nx6;
  int nx5;
  int nx4;
  bool is_sparse;
  bool is_vector;
  std::vector<std::string> component_labels;

  VarInfo() = delete;

  VarInfo(const std::string &label, const std::vector<std::string> &component_labels_,
          int vlen, int nx6, int nx5, int nx4, bool is_sparse, bool is_vector)
      : label(label), vlen(vlen), nx6(nx6), nx5(nx5), nx4(nx4), is_sparse(is_sparse),
        is_vector(is_vector) {
    if (vlen <= 0) {
      std::stringstream msg;
      msg << "### ERROR: Got variable " << label << " with length " << vlen
          << ". vlen must be greater than 0" << std::endl;
      PARTHENON_FAIL(msg);
    }

    // Note that this logic does not subscript components without component_labels if
    // there is only one component. Component names will be e.g.
    //   my_scalar
    // or
    //   my_non-vector_set_0
    //   my_non-vector_set_1
    // Note that this means the subscript will be dropped for multidim quantities if their
    // Nx6, Nx5, Nx4 are set to 1 at runtime e.g.
    //   my_non-vector_set
    // Similarly, if component labels are given for all components, those will be used
    // without the prefixed label.
    component_labels = {};
    if (vlen == 1 || is_vector) {
      component_labels = component_labels_.size() > 0 ? component_labels_
                                                      : std::vector<std::string>({label});
    } else if (component_labels_.size() == vlen) {
      component_labels = component_labels_;
    } else {
      for (int i = 0; i < vlen; i++) {
        component_labels.push_back(label + "_" + std::to_string(i));
      }
    }
  }

  explicit VarInfo(const std::shared_ptr<Variable<Real>> &var)
      : VarInfo(var->label(), var->metadata().getComponentLabels(), var->NumComponents(),
                var->GetDim(6), var->GetDim(5), var->GetDim(4), var->IsSparse(),
                var->IsSet(Metadata::Vector)) {}
};

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

static void writeXdmfSlabVariableRef(std::ofstream &fid, const std::string &name,
                                     const std::vector<std::string> &component_labels,
                                     std::string &hdfFile, int iblock, const int &vlen,
                                     int &ndims, hsize_t *dims,
                                     const std::string &dims321, bool isVector) {
  // writes a slab reference to file
  std::vector<std::string> names;
  int nentries = 1;
  if (vlen == 1 || isVector) {
    // we only make one entry, because either vlen == 1, or we write this as a vector
    names.push_back(name);
  } else {
    nentries = vlen;
    for (int i = 0; i < vlen; i++) {
      names.push_back(component_labels[i]);
    }
  }
  const int vector_size = isVector ? vlen : 1;

  const std::string prefix = "      ";
  for (int i = 0; i < nentries; i++) {
    fid << prefix << R"(<Attribute Name=")" << names[i] << R"(" Center="Cell")";
    if (isVector) {
      fid << R"( AttributeType="Vector")"
          << R"( Dimensions=")" << vector_size << " " << dims321 << R"(")";
    }
    fid << ">" << std::endl;
    fid << prefix << "  "
        << R"(<DataItem ItemType="HyperSlab" Dimensions=")" << dims321 << " "
        << vector_size << R"(">)" << std::endl;
    // TODO(JL) 3 and 5 are literals here, careful if they change.
    // "3" rows for START, STRIDE, and COUNT for each slab with "5" (H5_NDIM) entries.
    // START: iblock variable(_component)  0   0   0
    // STRIDE: 1               1           1   1   1
    // COUNT:  1           vector_size    nx3 nx2 nx1
    fid << prefix << "    "
        << R"(<DataItem Dimensions="3 5" NumberType="Int" Format="XML">)" << iblock << " "
        << i << " 0 0 0 "
        << " 1 1 1 1 1 1 " << vector_size << " " << dims321 << "</DataItem>" << std::endl;
    writeXdmfArrayRef(fid, prefix + "    ", hdfFile + ":/", name, dims, ndims, "Float",
                      8);
    fid << prefix << "  "
        << "</DataItem>" << std::endl;
    fid << prefix << "</Attribute>" << std::endl;
  }
}

void genXDMF(std::string hdfFile, Mesh *pm, SimTime *tm, int nx1, int nx2, int nx3,
             const std::vector<VarInfo> &var_list) {
  // using round robin generation.
  // must switch to MPIIO at some point

  // only rank 0 writes XDMF
  if (Globals::my_rank != 0) {
    return;
  }
  std::string filename_aux = hdfFile + ".xdmf";
  std::ofstream xdmf;
  hsize_t dims[H5_NDIM] = {0, 0, 0, 0, 0};

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

  int ndims = H5_NDIM;

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
    dims[1] = 1;
    dims[2] = nx3;
    dims[3] = nx2;
    dims[4] = nx1;
    for (const auto &vinfo : var_list) {
      const int vlen = vinfo.vlen;
      dims[1] = vlen;
      writeXdmfSlabVariableRef(xdmf, vinfo.label, vinfo.component_labels, hdfFile, ib,
                               vlen, ndims, dims, dims321, vinfo.is_vector);
    }
    xdmf << "      </Grid>" << std::endl;
  }
  xdmf << "    </Grid>" << std::endl;
  xdmf << "  </Domain>" << std::endl;
  xdmf << "</Xdmf>" << std::endl;
  xdmf.close();
}

void PHDF5Output::WriteOutputFile(Mesh *pm, ParameterInput *pin, SimTime *tm,
                                  const SignalHandler::OutputSignal signal) {
  if (output_params.single_precision_output) {
    this->template WriteOutputFileImpl<true>(pm, pin, tm, signal);
  } else {
    this->template WriteOutputFileImpl<false>(pm, pin, tm, signal);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void PHDF5Output:::WriteOutputFileImpl(Mesh *pm, ParameterInput *pin, bool flag)
//  \brief Cycles over all MeshBlocks and writes OutputData in the Parthenon HDF5 format,
//         one file per output using parallel IO.
template <bool WRITE_SINGLE_PRECISION>
void PHDF5Output::WriteOutputFileImpl(Mesh *pm, ParameterInput *pin, SimTime *tm,
                                      const SignalHandler::OutputSignal signal) {
  // writes all graphics variables to hdf file
  // HDF5 structures
  // Also writes companion xdmf file

  const int max_blocks_global = pm->nbtotal;
  const int num_blocks_local = static_cast<int>(pm->block_list.size());

  const IndexDomain theDomain =
      (output_params.include_ghost_zones ? IndexDomain::entire : IndexDomain::interior);

  auto const &first_block = *(pm->block_list.front());

  // shooting a blank just for getting the variable names
  const IndexRange out_ib = first_block.cellbounds.GetBoundsI(theDomain);
  const IndexRange out_jb = first_block.cellbounds.GetBoundsJ(theDomain);
  const IndexRange out_kb = first_block.cellbounds.GetBoundsK(theDomain);

  auto const nx1 = out_ib.e - out_ib.s + 1;
  auto const nx2 = out_jb.e - out_jb.s + 1;
  auto const nx3 = out_kb.e - out_kb.s + 1;

  const int rootLevel = pm->GetRootLevel();
  const int max_level = pm->GetCurrentLevel() - rootLevel;
  const auto &nblist = pm->GetNbList();

  // open HDF5 file
  // Define output filename
  auto filename = std::string(output_params.file_basename);
  filename.append(".");
  filename.append(output_params.file_id);
  filename.append(".");
  if (signal == SignalHandler::OutputSignal::now) {
    filename.append("now");
  } else if (signal == SignalHandler::OutputSignal::final &&
             output_params.file_label_final) {
    filename.append("final");
    // default time based data dump
  } else {
    std::stringstream file_number;
    file_number << std::setw(output_params.file_number_width) << std::setfill('0')
                << output_params.file_number;
    filename.append(file_number.str());
  }
  filename.append(restart_ ? ".rhdf" : ".phdf");

  if (signal == SignalHandler::OutputSignal::none) {
    // After file has been opened with the current number, already advance output
    // parameters so that for restarts the file is not immediatly overwritten again.
    // Only applies to default time-based data dumps, so that writing "now" and "final"
    // outputs does not change the desired output numbering.
    output_params.file_number++;
    output_params.next_time += output_params.dt;
    pin->SetInteger(output_params.block_name, "file_number", output_params.file_number);
    pin->SetReal(output_params.block_name, "next_time", output_params.next_time);
  }

  // set file access property list
#ifdef MPI_PARALLEL
  /* set the file access template for parallel IO access */
  H5P const acc_file = H5P::FromHIDCheck(H5Pcreate(H5P_FILE_ACCESS));

  /* ---------------------------------------------------------------------
     platform dependent code goes here -- the access template must be
     tuned for a particular filesystem blocksize.  some of these
     numbers are guesses / experiments, others come from the file system
     documentation.

     ---------------------------------------------------------------------- */

  // use collective metadata optimizations
#if H5_VERSION_GE(1, 10, 0)
  PARTHENON_HDF5_CHECK(H5Pset_coll_metadata_write(acc_file, true));
  PARTHENON_HDF5_CHECK(H5Pset_all_coll_metadata_ops(acc_file, true));
#endif

  bool exists, exists2;

  // Set the HDF5 format versions used when creating objects
  // Note, introducing API calls that create objects or features that are
  // only available to versions of the library greater than 1.8.x release will fail.
  // For that case, the highest version value will need to be increased.
  H5Pset_libver_bounds(acc_file, H5F_LIBVER_V18, H5F_LIBVER_V18);

  // Sets the maximum size of the data sieve buffer, in bytes.
  // The sieve_buf_size should be equal to a multiple of the disk block size
  // Default: Disabled
  size_t sieve_buf_size = Env::get<size_t>("H5_sieve_buf_size", 256 * KiB, exists);
  if (exists) {
    PARTHENON_HDF5_CHECK(H5Pset_sieve_buf_size(acc_file, sieve_buf_size));
  }

  // Sets the minimum metadata block size, in bytes.
  // Default: Disabled
  hsize_t meta_block_size = Env::get<hsize_t>("H5_meta_block_size", 8 * MiB, exists);
  if (exists) {
    PARTHENON_HDF5_CHECK(H5Pset_meta_block_size(acc_file, meta_block_size));
  }

  // Sets alignment properties of a file access property list.
  // Choose an alignment which is a multiple of the disk block size.
  // Default: Disabled
  hsize_t threshold; // Threshold value. Setting to 0 forces everything to be aligned.
  hsize_t alignment; // Alignment value.

  threshold = Env::get<hsize_t>("H5_alignment_threshold", 0, exists);
  alignment = Env::get<hsize_t>("H5_alignment_alignment", 8 * MiB, exists2);
  if (exists || exists2) {
    PARTHENON_HDF5_CHECK(H5Pset_alignment(acc_file, threshold, alignment));
  }

  // Defer metadata flush
  // Default: Disabled
  bool defer_metadata_flush = Env::get<bool>("H5_defer_metadata_flush", false, exists);
  if (defer_metadata_flush) {
    H5AC_cache_config_t cache_config;
    cache_config.version = H5AC__CURR_CACHE_CONFIG_VERSION;
    PARTHENON_HDF5_CHECK(H5Pget_mdc_config(acc_file, &cache_config));
    cache_config.set_initial_size = 1;
    cache_config.initial_size = 16 * MiB;
    cache_config.evictions_enabled = 0;
    cache_config.incr_mode = H5C_incr__off;
    cache_config.flash_incr_mode = H5C_flash_incr__off;
    cache_config.decr_mode = H5C_decr__off;
    PARTHENON_HDF5_CHECK(H5Pset_mdc_config(acc_file, &cache_config));
  }

  /* create an MPI_INFO object -- on some platforms it is useful to
     pass some information onto the underlying MPI_File_open call */
  MPI_Info FILE_INFO_TEMPLATE;
  PARTHENON_MPI_CHECK(MPI_Info_create(&FILE_INFO_TEMPLATE));

  // Free MPI_Info on error on return or throw
  struct MPI_InfoDeleter {
    MPI_Info info;
    ~MPI_InfoDeleter() { MPI_Info_free(&info); }
  } delete_info{FILE_INFO_TEMPLATE};

  // Hint specifies the manner in which the file will be accessed until the file is closed
  const auto access_style =
      Env::get<std::string>("MPI_access_style", "write_once", exists);
  PARTHENON_MPI_CHECK(
      MPI_Info_set(FILE_INFO_TEMPLATE, "access_style", access_style.c_str()));

  // Specifies whether the application may benefit from collective buffering
  // Default :: collective_buffering is disabled
  bool collective_buffering = Env::get<bool>("MPI_collective_buffering", false, exists);
  if (exists) {
    PARTHENON_MPI_CHECK(MPI_Info_set(FILE_INFO_TEMPLATE, "collective_buffering", "true"));
    // Specifies the block size to be used for collective buffering file acces
    const auto cb_block_size =
        Env::get<std::string>("MPI_cb_block_size", "1048576", exists);
    PARTHENON_MPI_CHECK(
        MPI_Info_set(FILE_INFO_TEMPLATE, "cb_block_size", cb_block_size.c_str()));
    // Specifies the total buffer space that can be used for collective buffering on each
    // target node, usually a multiple of cb_block_size
    const auto cb_buffer_size =
        Env::get<std::string>("MPI_cb_buffer_size", "4194304", exists);
    PARTHENON_MPI_CHECK(
        MPI_Info_set(FILE_INFO_TEMPLATE, "cb_buffer_size", cb_buffer_size.c_str()));
  }

  /* tell the HDF5 library that we want to use MPI-IO to do the writing */
  PARTHENON_HDF5_CHECK(H5Pset_fapl_mpio(acc_file, MPI_COMM_WORLD, FILE_INFO_TEMPLATE));
#else
  hid_t const acc_file = H5P_DEFAULT;
#endif // ifdef MPI_PARALLEL

  // now create the file
  H5F file;
  try {
    file = H5F::FromHIDCheck(
        H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, acc_file));
  } catch (std::exception &ex) {
    std::stringstream err;
    err << "### ERROR: Failed to create HDF5 output file '" << filename
        << "' with the following error:" << std::endl
        << ex.what() << std::endl;
    PARTHENON_THROW(err)
  }

  // -------------------------------------------------------------------------------- //
  //   WRITING ATTRIBUTES                                                             //
  // -------------------------------------------------------------------------------- //
  {
    // write input key-value pairs
    std::ostringstream oss;
    pin->ParameterDump(oss);

    // Mesh information
    const H5G input_group = MakeGroup(file, "/Input");

    HDF5WriteAttribute("File", oss.str().c_str(), input_group);
  } // Input section

  // we'll need this again at the end
  const H5G info_group = MakeGroup(file, "/Info");
  {
    HDF5WriteAttribute("OutputFormatVersion", OUTPUT_VERSION_FORMAT, info_group);

    if (tm != nullptr) {
      HDF5WriteAttribute("NCycle", tm->ncycle, info_group);
      HDF5WriteAttribute("Time", tm->time, info_group);
      HDF5WriteAttribute("dt", tm->dt, info_group);
    }

    HDF5WriteAttribute("WallTime", Driver::elapsed_main(), info_group);
    HDF5WriteAttribute("NumDims", pm->ndim, info_group);
    HDF5WriteAttribute("NumMeshBlocks", pm->nbtotal, info_group);
    HDF5WriteAttribute("MaxLevel", max_level, info_group);
    // write whether we include ghost cells or not
    HDF5WriteAttribute("IncludesGhost", output_params.include_ghost_zones ? 1 : 0,
                       info_group);
    // write number of ghost cells in simulation
    HDF5WriteAttribute("NGhost", Globals::nghost, info_group);
    HDF5WriteAttribute("Coordinates", std::string(first_block.coords.Name()).c_str(),
                       info_group);

    // restart info, write always
    HDF5WriteAttribute("NBNew", pm->nbnew, info_group);
    HDF5WriteAttribute("NBDel", pm->nbdel, info_group);
    HDF5WriteAttribute("RootLevel", rootLevel, info_group);
    HDF5WriteAttribute("Refine", pm->adaptive ? 1 : 0, info_group);
    HDF5WriteAttribute("Multilevel", pm->multilevel ? 1 : 0, info_group);

    HDF5WriteAttribute("BlocksPerPE", nblist, info_group);

    // Mesh block size
    HDF5WriteAttribute("MeshBlockSize", std::vector<int>{nx1, nx2, nx3}, info_group);

    // RootGridDomain - float[9] array with xyz mins, maxs, rats (dx(i)/dx(i-1))
    HDF5WriteAttribute(
        "RootGridDomain",
        std::vector<Real>{pm->mesh_size.x1min, pm->mesh_size.x1max, pm->mesh_size.x1rat,
                          pm->mesh_size.x2min, pm->mesh_size.x2max, pm->mesh_size.x2rat,
                          pm->mesh_size.x3min, pm->mesh_size.x3max, pm->mesh_size.x3rat},
        info_group);

    // Root grid size (number of cells at root level)
    HDF5WriteAttribute(
        "RootGridSize",
        std::vector<int>{pm->mesh_size.nx1, pm->mesh_size.nx2, pm->mesh_size.nx3},
        info_group);

    // Boundary conditions
    std::vector<std::string> boundary_condition_str(BOUNDARY_NFACES);
    for (size_t i = 0; i < boundary_condition_str.size(); i++) {
      boundary_condition_str[i] = GetBoundaryString(pm->mesh_bcs[i]);
    }

    HDF5WriteAttribute("BoundaryConditions", boundary_condition_str, info_group);
  } // Info section

  // write Params
  {
    const H5G params_group = MakeGroup(file, "/Params");

    for (const auto &package : pm->packages.AllPackages()) {
      const auto state = package.second;
      // Write all params that can be written as HDF5 attributes
      state->AllParams().WriteAllToHDF5(state->label(), params_group);
    }
  } // Params section

  // -------------------------------------------------------------------------------- //
  //   WRITING MESHBLOCK METADATA                                                     //
  // -------------------------------------------------------------------------------- //

  // set local offset, always the same for all data sets
  hsize_t my_offset = 0;
  for (int i = 0; i < Globals::my_rank; i++) {
    my_offset += nblist[i];
  }

  const std::array<hsize_t, H5_NDIM> local_offset({my_offset, 0, 0, 0, 0});

  // these can vary by data set, except index 0 is always the same
  std::array<hsize_t, H5_NDIM> local_count(
      {static_cast<hsize_t>(num_blocks_local), 1, 1, 1, 1});
  std::array<hsize_t, H5_NDIM> global_count(
      {static_cast<hsize_t>(max_blocks_global), 1, 1, 1, 1});

  // for convenience
  const hsize_t *const p_loc_offset = local_offset.data();
  const hsize_t *const p_loc_cnt = local_count.data();
  const hsize_t *const p_glob_cnt = global_count.data();

  H5P const pl_xfer = H5P::FromHIDCheck(H5Pcreate(H5P_DATASET_XFER));
  H5P const pl_dcreate = H5P::FromHIDCheck(H5Pcreate(H5P_DATASET_CREATE));

  // Never write fill values to the dataset
  PARTHENON_HDF5_CHECK(H5Pset_fill_time(pl_dcreate, H5D_FILL_TIME_NEVER));

#ifndef PARTHENON_DISABLE_HDF5_COMPRESSION
  if (output_params.hdf5_compression_level > 0) {
    // we need chunks to enable compression
    const std::array<hsize_t, H5_NDIM> chunk_size({1, 1, static_cast<hsize_t>(nx3),
                                                   static_cast<hsize_t>(nx2),
                                                   static_cast<hsize_t>(nx1)});
    PARTHENON_HDF5_CHECK(H5Pset_chunk(pl_dcreate, H5_NDIM, chunk_size.data()));
    PARTHENON_HDF5_CHECK(
        H5Pset_deflate(pl_dcreate, std::min(9, output_params.hdf5_compression_level)));
  }
#endif

#ifdef MPI_PARALLEL
  PARTHENON_HDF5_CHECK(H5Pset_dxpl_mpio(pl_xfer, H5FD_MPIO_COLLECTIVE));
#endif

  // write Blocks metadata
  {
    const H5G gBlocks = MakeGroup(file, "/Blocks");

    // write Xmin[ndim] for blocks
    {
      std::vector<Real> tmpData(num_blocks_local * 3);
      int i = 0;

      for (auto &pmb : pm->block_list) {
        auto xmin = pmb->coords.GetXmin();
        tmpData[i++] = xmin[0];
        if (pm->ndim > 1) {
          tmpData[i++] = xmin[1];
        }
        if (pm->ndim > 2) {
          tmpData[i++] = xmin[2];
        }
      }
      local_count[1] = global_count[1] = pm->ndim;
      HDF5Write2D(gBlocks, "xmin", tmpData.data(), p_loc_offset, p_loc_cnt, p_glob_cnt,
                  pl_xfer);
    }

    // write Block ID
    {
      // LOC.lx1,2,3
      hsize_t n = 3;
      std::vector<int64_t> tmpLoc(num_blocks_local * n);
      local_count[1] = global_count[1] = n;

      int i = 0;
      for (auto &pmb : pm->block_list) {
        tmpLoc[i++] = pmb->loc.lx1;
        tmpLoc[i++] = pmb->loc.lx2;
        tmpLoc[i++] = pmb->loc.lx3;
      }
      HDF5Write2D(gBlocks, "loc.lx123", tmpLoc.data(), p_loc_offset, p_loc_cnt,
                  p_glob_cnt, pl_xfer);

      // (LOC.)level, GID, LID, cnghost, gflag
      n = 5; // this is NOT H5_NDIM
      std::vector<int> tmpID(num_blocks_local * n);
      local_count[1] = global_count[1] = n;

      i = 0;
      for (auto &pmb : pm->block_list) {
        tmpID[i++] = pmb->loc.level;
        tmpID[i++] = pmb->gid;
        tmpID[i++] = pmb->lid;
        tmpID[i++] = pmb->cnghost;
        tmpID[i++] = pmb->gflag;
      }
      HDF5Write2D(gBlocks, "loc.level-gid-lid-cnghost-gflag", tmpID.data(), p_loc_offset,
                  p_loc_cnt, p_glob_cnt, pl_xfer);
    }
  } // Block section

  // Write mesh coordinates to file
  for (const bool face : {true, false}) {
    const H5G gLocations = MakeGroup(file, face ? "/Locations" : "/VolumeLocations");
    const int offset = face ? 1 : 0;

    // write X coordinates
    std::vector<Real> loc_x((nx1 + offset) * num_blocks_local);
    std::vector<Real> loc_y((nx2 + offset) * num_blocks_local);
    std::vector<Real> loc_z((nx3 + offset) * num_blocks_local);

    size_t idx_x = 0;
    size_t idx_y = 0;
    size_t idx_z = 0;

    for (size_t b = 0; b < pm->block_list.size(); ++b) {
      auto &pmb = pm->block_list[b];

      for (int i = out_ib.s; i <= out_ib.e + offset; ++i) {
        loc_x[idx_x++] = face ? pmb->coords.x1f(0, 0, i) : pmb->coords.x1v(0, 0, i);
      }

      for (int j = out_jb.s; j <= out_jb.e + offset; ++j) {
        loc_y[idx_y++] = face ? pmb->coords.x2f(0, j, 0) : pmb->coords.x2v(0, j, 0);
      }

      for (int k = out_kb.s; k <= out_kb.e + offset; ++k) {
        loc_z[idx_z++] = face ? pmb->coords.x3f(k, 0, 0) : pmb->coords.x3v(k, 0, 0);
      }
    }

    local_count[1] = global_count[1] = nx1 + offset;
    HDF5Write2D(gLocations, "x", loc_x.data(), p_loc_offset, p_loc_cnt, p_glob_cnt,
                pl_xfer);

    local_count[1] = global_count[1] = nx2 + offset;
    HDF5Write2D(gLocations, "y", loc_y.data(), p_loc_offset, p_loc_cnt, p_glob_cnt,
                pl_xfer);

    local_count[1] = global_count[1] = nx3 + offset;
    HDF5Write2D(gLocations, "z", loc_z.data(), p_loc_offset, p_loc_cnt, p_glob_cnt,
                pl_xfer);
  }

  // Write Levels and Logical Locations with the level for each Meshblock loclist contains
  // levels and logical locations for all meshblocks on all ranks
  {
    const auto &loclist = pm->GetLocList();

    std::vector<std::int64_t> levels;
    levels.reserve(pm->nbtotal);

    std::vector<std::int64_t> logicalLocations;
    logicalLocations.reserve(pm->nbtotal * 3);

    for (const auto &loc : loclist) {
      levels.push_back(loc.level - pm->GetRootLevel());
      logicalLocations.push_back(loc.lx1);
      logicalLocations.push_back(loc.lx2);
      logicalLocations.push_back(loc.lx3);
    }

    // Only write levels on rank 0 since it has data for all ranks
    local_count[0] = (Globals::my_rank == 0) ? pm->nbtotal : 0;
    HDF5WriteND(file, "Levels", levels.data(), 1, local_offset.data(), local_count.data(),
                global_count.data(), pl_xfer, H5P_DEFAULT);

    local_count[1] = global_count[1] = 3;
    HDF5Write2D(file, "LogicalLocations", logicalLocations.data(), local_offset.data(),
                local_count.data(), global_count.data(), pl_xfer);

    // reset for collective output
    local_count[0] = num_blocks_local;
  }

  // -------------------------------------------------------------------------------- //
  //   WRITING VARIABLES DATA                                                         //
  // -------------------------------------------------------------------------------- //

  // All blocks have the same list of variable metadata that exist in the entire
  // simulation, but not all variables may be allocated on all blocks

  auto get_vars = [=](const std::shared_ptr<MeshBlock> pmb) {
    if (restart_) {
      // get all vars with flag Independent OR restart
      return pmb->meshblock_data.Get()
          ->GetVariablesByFlag(
              {parthenon::Metadata::Independent, parthenon::Metadata::Restart}, false)
          .vars();
    } else {
      return pmb->meshblock_data.Get()
          ->GetVariablesByName(output_params.variables)
          .vars();
    }
  };

  // get list of all vars, just use the first block since the list is the same for all
  // blocks
  std::vector<VarInfo> all_vars_info;
  const auto vars = get_vars(pm->block_list.front());
  for (auto &v : vars) {
    all_vars_info.emplace_back(v);
  }

  // sort alphabetically
  std::sort(all_vars_info.begin(), all_vars_info.end(),
            [](const VarInfo &a, const VarInfo &b) { return a.label < b.label; });

  // We need to add information about the sparse variables to the HDF5 file, namely:
  // 1) Which variables are sparse
  // 2) Is a sparse id of a particular sparse variable allocated on a given block
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
  // v is allocated on the block with index b, otherwise the value is false

  std::vector<std::string> sparse_names;
  std::unordered_map<std::string, size_t> sparse_field_idx;
  for (auto &vinfo : all_vars_info) {
    if (vinfo.is_sparse) {
      sparse_field_idx.insert({vinfo.label, sparse_names.size()});
      sparse_names.push_back(vinfo.label);
    }
  }

  hsize_t num_sparse = sparse_names.size();
  // can't use std::vector here because std::vector<hbool_t> is the same as
  // std::vector<bool> and it doesn't have .data() member
  std::unique_ptr<hbool_t[]> sparse_allocated(new hbool_t[num_blocks_local * num_sparse]);

  // allocate space for largest size variable
  const hsize_t varSize = nx3 * nx2 * nx1;
  int vlen_max = 0;
  for (auto &vinfo : all_vars_info) {
    vlen_max = std::max(vlen_max, vinfo.vlen);
  }

  using OutT = typename std::conditional<WRITE_SINGLE_PRECISION, float, Real>::type;
  std::vector<OutT> tmpData(varSize * vlen_max * num_blocks_local);

  // create persistent spaces
  local_count[0] = num_blocks_local;
  global_count[0] = max_blocks_global;
  local_count[2] = global_count[2] = nx3;
  local_count[3] = global_count[3] = nx2;
  local_count[4] = global_count[4] = nx1;

  // for each variable we write
  for (auto &vinfo : all_vars_info) {
    // not really necessary, but doesn't hurt
    memset(tmpData.data(), 0, tmpData.size() * sizeof(OutT));

    const std::string var_name = vinfo.label;
    const hsize_t vlen = vinfo.vlen;
    const hsize_t nx6 = vinfo.nx6;
    const hsize_t nx5 = vinfo.nx5;
    const hsize_t nx4 = vinfo.nx4;

    local_count[1] = global_count[1] = vlen;

    // load up data
    hsize_t index = 0;

    // for each local mesh block
    for (size_t b_idx = 0; b_idx < num_blocks_local; ++b_idx) {
      const auto &pmb = pm->block_list[b_idx];
      bool is_allocated = false;

      // for each variable that this local meshblock actually has
      const auto vars = get_vars(pmb);
      for (auto &v : vars) {
        // For reference, if we update the logic here, there's also
        // a similar block in parthenon_manager.cpp
        if (v->IsAllocated() && (var_name == v->label())) {
          auto v_h = v->data.GetHostMirrorAndCopy();
          for (int t = 0; t < nx6; ++t) {
            for (int u = 0; u < nx5; ++u) {
              for (int v = 0; v < nx4; ++v) {
                for (int k = out_kb.s; k <= out_kb.e; ++k) {
                  for (int j = out_jb.s; j <= out_jb.e; ++j) {
                    for (int i = out_ib.s; i <= out_ib.e; ++i) {
                      tmpData[index++] = static_cast<OutT>(v_h(t, u, v, k, j, i));
                    }
                  }
                }
              }
            }
          }

          is_allocated = true;
          break;
        }
      }

      if (vinfo.is_sparse) {
        size_t sparse_idx = sparse_field_idx.at(vinfo.label);
        sparse_allocated[b_idx * num_sparse + sparse_idx] = is_allocated;
      }

      if (!is_allocated) {
        if (vinfo.is_sparse) {
          hsize_t N = varSize * vlen;
          memset(tmpData.data() + index, 0, N * sizeof(OutT));
          index += N;
        } else {
          std::stringstream msg;
          msg << "### ERROR: Unable to find dense variable " << var_name << std::endl;
          PARTHENON_FAIL(msg);
        }
      }
    }

    // write data to file
    HDF5WriteND(file, var_name, tmpData.data(), H5_NDIM, p_loc_offset, p_loc_cnt,
                p_glob_cnt, pl_xfer, pl_dcreate);
  }

  // names of variables
  std::vector<std::string> var_names;
  var_names.reserve(all_vars_info.size());

  // number of components within each dataset
  std::vector<size_t> num_components;
  num_components.reserve(all_vars_info.size());

  // names of components within each dataset
  std::vector<std::string> component_names;
  component_names.reserve(all_vars_info.size()); // may be larger

  for (const auto &vi : all_vars_info) {
    var_names.push_back(vi.label);

    const auto &component_labels = vi.component_labels;
    PARTHENON_REQUIRE_THROWS(component_labels.size() > 0, "Got 0 component labels");

    num_components.push_back(component_labels.size());
    for (const auto &label : component_labels) {
      component_names.push_back(label);
    }
  }

  HDF5WriteAttribute("NumComponents", num_components, info_group);
  HDF5WriteAttribute("ComponentNames", component_names, info_group);
  HDF5WriteAttribute("OutputDatasetNames", var_names, info_group);

  // write SparseInfo and SparseFields (we can't write a zero-size dataset, so only write
  // this if we have sparse fields)
  if (num_sparse > 0) {
    local_count[1] = global_count[1] = num_sparse;

    HDF5Write2D(file, "SparseInfo", sparse_allocated.get(), p_loc_offset, p_loc_cnt,
                p_glob_cnt, pl_xfer);

    // write names of sparse fields as attribute, first convert to vector of const char*
    std::vector<const char *> names(num_sparse);
    for (size_t i = 0; i < num_sparse; ++i)
      names[i] = sparse_names[i].c_str();

    const H5D dset = H5D::FromHIDCheck(H5Dopen2(file, "SparseInfo", H5P_DEFAULT));
    HDF5WriteAttribute("SparseFields", names, dset);
  } // SparseInfo and SparseFields sections

  // generate XDMF companion file
  genXDMF(filename, pm, tm, nx1, nx2, nx3, all_vars_info);
}

// explicit template instantiation
template void PHDF5Output::WriteOutputFileImpl<false>(Mesh *, ParameterInput *, SimTime *,
                                                      SignalHandler::OutputSignal);
template void PHDF5Output::WriteOutputFileImpl<true>(Mesh *, ParameterInput *, SimTime *,
                                                     SignalHandler::OutputSignal);

} // namespace parthenon

#endif // ifdef ENABLE_HDF5
