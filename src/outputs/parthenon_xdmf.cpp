//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2023 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020-2023. Triad National Security, LLC. All rights reserved.
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

// only proceed if HDF5 output enabled
#ifdef ENABLE_HDF5

// HDF5
#include <hdf5.h>

// C++
#include <iostream>
#include <string>

// Parthenon
#include "basic_types.hpp"
#include "mesh/mesh.hpp"
#include "outputs/output_utils.hpp"
#include "outputs/parthenon_hdf5.hpp"
#include "outputs/parthenon_xdmf.hpp"
#include "utils/utils.hpp"

#define PARTHENON_ENABLE_PARTICLE_XDMF 0

namespace parthenon {
using namespace OutputUtils;

namespace XDMF {
namespace impl {
// XDMF subroutine to write a dataitem that refers to an HDF array
static std::string stringXdmfArrayRef(const std::string &prefix,
                                      const std::string &hdfPath,
                                      const std::string &label, const hsize_t *dims,
                                      const int &ndims, const std::string &theType,
                                      const int &precision);
static void writeXdmfArrayRef(std::ofstream &fid, const std::string &prefix,
                              const std::string &hdfPath, const std::string &label,
                              const hsize_t *dims, const int &ndims,
                              const std::string &theType, const int &precision);
static void writeXdmfSlabVariableRef(std::ofstream &fid, const std::string &name,
                                     const std::vector<std::string> &component_labels,
                                     std::string &hdfFile, int iblock,
                                     const int &num_components, int &ndims, hsize_t *dims,
                                     const std::string &dims321, bool isVector);
static std::string ParticleDatasetRef(const std::string &prefix,
                                      const std::string &swmname,
                                      const std::string &varname,
                                      const std::string &hdffile,
                                      const std::string &datatype,
                                      const std::string &extradims, int particle_count);
static void ParticleVariableRef(std::ofstream &xdmf, const std::string &varname,
                                const SwarmVarInfo &varinfo, const std::string &swmname,
                                const std::string &hdffile, int particle_count);
} // namespace impl

void genXDMF(std::string hdfFile, Mesh *pm, SimTime *tm, int nx1, int nx2, int nx3,
             const std::vector<VarInfo> &var_list, const AllSwarmInfo &all_swarm_info) {
  using namespace HDF5;
  using namespace OutputUtils;
  using namespace impl;
  // using round robin generation.
  // must switch to MPIIO at some point

  // only rank 0 writes XDMF
  if (Globals::my_rank != 0) {
    return;
  }
  std::string filename_aux = hdfFile + ".xdmf";
  std::ofstream xdmf;
  hsize_t dims[H5_NDIM] = {0};

  // open file
  xdmf = std::ofstream(filename_aux.c_str(), std::ofstream::trunc);

  // Write header
  xdmf << R"(<?xml version="1.0" ?>)" << std::endl;
  xdmf << R"(<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd">)" << std::endl;
  xdmf << R"(<Xdmf Version="3.0">)" << std::endl;
  xdmf << R"(<Information Name="TimeVaryingMetaData" Value="True"/>)" << std::endl;
  xdmf << "  <Domain>" << std::endl;
  xdmf << R"(  <Grid Name="Mesh" GridType="Collection">)" << std::endl;
  if (tm != nullptr) {
    xdmf << R"(    <Information Name="Cycle" Value=")" << tm->ncycle << R"("/>)"
         << std::endl;
    xdmf << R"(    <Time Value=")" << tm->time << R"("/>)" << std::endl;
  }

  std::string blockTopology =
      R"(      <Topology TopologyType="3DRectMesh" Dimensions=")" +
      std::to_string(nx3 + 1) + " " + std::to_string(nx2 + 1) + " " +
      std::to_string(nx1 + 1) + R"("/>)" + '\n';
  const std::string slabPreDim = R"(        <DataItem ItemType="HyperSlab" Dimensions=")";
  const std::string slabPreBlock2D =
      R"("><DataItem Dimensions="3 2" NumberType="Int" Format="XML">)";
  const std::string slabTrailer = "</DataItem>";

  // Now write Grid for each block
  dims[0] = pm->nbtotal;
  std::string dims321 =
      std::to_string(nx3) + " " + std::to_string(nx2) + " " + std::to_string(nx1);

  for (int ib = 0; ib < pm->nbtotal; ib++) {
    xdmf << "    <Grid GridType=\"Uniform\" Name=\"" << ib << "\">" << std::endl;
    xdmf << blockTopology;
    xdmf << R"(      <Geometry GeometryType="VXVYVZ">)" << std::endl;
    xdmf << slabPreDim << nx1 + 1 << slabPreBlock2D << ib << " 0 1 1 1 " << nx1 + 1
         << slabTrailer << std::endl;

    dims[1] = nx1 + 1;
    writeXdmfArrayRef(xdmf, "          ", hdfFile + ":/Locations/", "x", dims, 2, "Float",
                      8);
    xdmf << "        </DataItem>" << std::endl;

    xdmf << slabPreDim << nx2 + 1 << slabPreBlock2D << ib << " 0 1 1 1 " << nx2 + 1
         << slabTrailer << std::endl;

    dims[1] = nx2 + 1;
    writeXdmfArrayRef(xdmf, "          ", hdfFile + ":/Locations/", "y", dims, 2, "Float",
                      8);
    xdmf << "        </DataItem>" << std::endl;

    xdmf << slabPreDim << nx3 + 1 << slabPreBlock2D << ib << " 0 1 1 1 " << nx3 + 1
         << slabTrailer << std::endl;

    dims[1] = nx3 + 1;
    writeXdmfArrayRef(xdmf, "          ", hdfFile + ":/Locations/", "z", dims, 2, "Float",
                      8);
    xdmf << "        </DataItem>" << std::endl;

    xdmf << "      </Geometry>" << std::endl;

    // write graphics variables
    int ndim;
    for (const auto &vinfo : var_list) {
      auto alldims = vinfo.GetShape<hsize_t>();
      // Only cell-based data currently supported for visualization
      if (vinfo.where == MetadataFlag(Metadata::Cell)) {
        ndim = 3 + vinfo.tensor_rank + 1;
        for (int i = 0; i < vinfo.tensor_rank; i++) {
          dims[1 + i] = alldims[3 - vinfo.tensor_rank + i];
        }
        dims[vinfo.tensor_rank + 1] = nx3;
        dims[vinfo.tensor_rank + 2] = nx2;
        dims[vinfo.tensor_rank + 3] = nx1;
      } else {
        continue;
      }

      const int num_components = vinfo.num_components;
      writeXdmfSlabVariableRef(xdmf, vinfo.label, vinfo.component_labels, hdfFile, ib,
                               num_components, ndim, dims, dims321, vinfo.is_vector);
    }
    xdmf << "    </Grid>" << std::endl;
  }
  // Particles are defined as their own "mesh"
#if PARTHENON_ENABLE_PARTICLE_XDMF
  for (const auto &[swmname, swminfo] : all_swarm_info.all_info) {
    xdmf << StringPrintf(
        "    <Grid GridType=\"Uniform\" Name=\"%s\">\n"
        "      <Topology TopologyType=\"Polyvertex\" Dimensions=\"%d\" "
        "NodesPerElement=\"1\">\n"
        "        <DataItem Format=\"HDF\" Dimensions=\"%d\" NumberType=\"Int\">\n"
        "          %s:/%s/SwarmVars/id\n"
        "        </DataItem>\n"
        "      </Topology>\n"
        "      <Geometry GeometryType=\"VXVYVZ\">\n",
        swmname.c_str(), swminfo.global_count, swminfo.global_count, hdfFile.c_str(),
        swmname.c_str());
    xdmf << ParticleDatasetRef("        ", swmname, "x", hdfFile, "Float", "",
                               swminfo.global_count);
    xdmf << ParticleDatasetRef("        ", swmname, "y", hdfFile, "Float", "",
                               swminfo.global_count);
    xdmf << ParticleDatasetRef("        ", swmname, "z", hdfFile, "Float", "",
                               swminfo.global_count);
    xdmf << "      </Geometry>" << std::endl;
    for (const auto &[varname, varinfo] : swminfo.var_info) {
      if ((varname == "id") || (varname == "x") || (varname == "y") || (varname == "z")) {
        continue; // We already did this one!
      }
      ParticleVariableRef(xdmf, varname, varinfo, swmname, hdfFile, swminfo.global_count);
    }
    xdmf << "    </Grid>" << std::endl;
  }
#endif

  // Cleanup
  xdmf << "    </Grid>" << std::endl;
  xdmf << "  </Domain>" << std::endl;
  xdmf << "</Xdmf>" << std::endl;
  xdmf.close();
}

namespace impl {
// XDMF subroutine to write a dataitem that refers to an HDF array
static std::string stringXdmfArrayRef(const std::string &prefix,
                                      const std::string &hdfPath,
                                      const std::string &label, const hsize_t *dims,
                                      const int &ndims, const std::string &theType,
                                      const int &precision) {
  std::string mystr = prefix + R"(<DataItem Format="HDF" Dimensions=")";
  for (int i = 0; i < ndims; i++) {
    mystr += " " + std::to_string(dims[i]);
  }
  mystr += "\" Name=\"" + label + "\"";
  mystr += " NumberType=\"" + theType + "\"";
  mystr += R"( Precision=")" + std::to_string(precision) + R"(">)" + '\n';
  mystr += prefix + "  " + hdfPath + label + '\n' + prefix + "</DataItem>" + '\n';
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
                                     std::string &hdfFile, int iblock,
                                     const int &num_components, int &ndims, hsize_t *dims,
                                     const std::string &dims321, bool isVector) {
  // writes a slab reference to file
  std::vector<std::string> names;
  int nentries = 1;
  if (num_components == 1 || isVector) {
    // we only make one entry, because either num_components == 1, or we write this as a
    // vector
    names.push_back(name);
  } else {
    nentries = num_components;
    for (int i = 0; i < num_components; i++) {
      names.push_back(component_labels[i]);
    }
  }
  const int tensor_dims = ndims - 1 - 3;

  if (tensor_dims == 0) {
    const std::string prefix = "      ";
    fid << prefix << R"(<Attribute Name=")" << names[0] << R"(" Center="Cell")";
    fid << ">" << std::endl;
    fid << prefix << "  "
        << R"(<DataItem ItemType="HyperSlab" Dimensions=")";
    fid << dims321 << " ";
    fid << R"(">)" << std::endl;
    // "3" rows for START, STRIDE, and COUNT for each slab with "4" entries.
    // START: iblock 0   0   0
    // STRIDE: 1     1   1   1
    // COUNT:  1     nx3 nx2 nx1
    fid << prefix << "    "
        << R"(<DataItem Dimensions="3 4" NumberType="Int" Format="XML">)" << iblock << " "
        << " 0 0 0 "
        << " 1 1 1 1 1 "
        << " " << dims321 << "</DataItem>" << std::endl;
    writeXdmfArrayRef(fid, prefix + "    ", hdfFile + ":/", name, dims, ndims, "Float",
                      8);
    fid << prefix << "  "
        << "</DataItem>" << std::endl;
    fid << prefix << "</Attribute>" << std::endl;
  } else if (tensor_dims == 1) {
    const std::string prefix = "      ";
    for (int i = 0; i < nentries; i++) {
      fid << prefix << R"(<Attribute Name=")" << names[i] << R"(" Center="Cell")";
      if (isVector) {
        fid << R"( AttributeType="Vector")"
            << R"( Dimensions=")" << dims[1] << " " << dims321 << R"(")";
      }
      fid << ">" << std::endl;
      fid << prefix << "  "
          << R"(<DataItem ItemType="HyperSlab" Dimensions=")";
      fid << dims321 << " ";
      fid << R"(">)" << std::endl;
      // "3" rows for START, STRIDE, and COUNT for each slab with "5" entries.
      // START: iblock variable(_component)  0   0   0
      // STRIDE: 1               1           1   1   1
      // COUNT:  1               dims[1]     nx3 nx2 nx1
      fid << prefix << "    "
          << R"(<DataItem Dimensions="3 5" NumberType="Int" Format="XML">)" << iblock
          << " " << i << " 0 0 0 "
          << " 1 1 1 1 1 1 1"
          << " " << dims321 << "</DataItem>" << std::endl;
      writeXdmfArrayRef(fid, prefix + "    ", hdfFile + ":/", name, dims, ndims, "Float",
                        8);
      fid << prefix << "  "
          << "</DataItem>" << std::endl;
      fid << prefix << "</Attribute>" << std::endl;
    }
  }
  // TODO(BRR) Support tensor dims 2 and 3
}
static std::string ParticleDatasetRef(const std::string &prefix,
                                      const std::string &swmname,
                                      const std::string &varname,
                                      const std::string &hdffile,
                                      const std::string &datatype,
                                      const std::string &extradims, int particle_count) {
  std::string precision_string = (datatype == "Float") ? " Precision=\"8\"" : "";
  auto part =
      StringPrintf("%s<DataItem Format=\"HDF\" Dimensions=\"%s%d\" Name=\"%s\" "
                   "NumberType=\"%s\"%s>\n"
                   "%s  %s:/%s/SwarmVars/%s\n"
                   "%s</DataItem>\n",
                   prefix.c_str(), extradims.c_str(), particle_count, varname.c_str(),
                   datatype.c_str(), precision_string.c_str(), prefix.c_str(),
                   hdffile.c_str(), swmname.c_str(), varname.c_str(), prefix.c_str());
  return part;
}
static void ParticleVariableRef(std::ofstream &xdmf, const std::string &varname,
                                const SwarmVarInfo &varinfo, const std::string &swmname,
                                const std::string &hdffile, int particle_count) {
  const char fmt[] =
      "      <Attribute Name=\"%s\" AttributeType=\"%s\" Center=\"Node\">\n";
  if (varinfo.vector) { // vector logic
    std::string name = swmname + "/" + varname;
    xdmf << StringPrintf(fmt, name.c_str(), "Vector");
    xdmf << StringPrintf("        <DataItem Dimensions=\"%d 3\" Function=\"JOIN($0, $1, "
                         "$2)\" ItemType=\"Function\">\n",
                         particle_count);
    for (int d = 0; d < 3; ++d) {
      xdmf << StringPrintf(
          "          <DataItem ItemType=\"HyperSlab\" Dimensions=\"%d\">\n"
          "            <DataItem Dimensions=\"3 2\" Format=\"XML\">\n"
          "              %d 0\n"
          "              1 1\n"
          "              1 %d\n"
          "            </DataItem>\n",
          particle_count, d, particle_count);
      xdmf << ParticleDatasetRef("            ", swmname, varname, hdffile,
                                 varinfo.swtype, "3 ", particle_count);
      xdmf << "          </DataItem>" << std::endl;
    }
    xdmf << "        </DataItem>" << std::endl;
    xdmf << "      </Attribute>" << std::endl;
  } else {
    const int rank = varinfo.tensor_rank;
    std::string extradims;
    for (int i = 6; i >= 2; --i) {
      extradims += StringPrintf("%d ", varinfo.GetN(i));
    }
    for (int n6 = 0; n6 < varinfo.GetN(6); ++n6) {
      for (int n5 = 0; n5 < varinfo.GetN(5); ++n5) {
        for (int n4 = 0; n4 < varinfo.GetN(4); ++n4) {
          for (int n3 = 0; n3 < varinfo.GetN(3); ++n3) {
            for (int n2 = 0; n2 < varinfo.GetN(2); ++n2) {
              std::string name = swmname + "/" + varname;
              if (rank > 4) name += StringPrintf("_%03d", n6);
              if (rank > 3) name += StringPrintf("_%03d", n5);
              if (rank > 2) name += StringPrintf("_%03d", n4);
              if (rank > 1) name += StringPrintf("_%03d", n3);
              if (rank > 0) name += StringPrintf("_%03d", n2);
              xdmf << StringPrintf(fmt, name.c_str(), "Scalar");
              if (rank > 0) {
                std::string starts = "";
                int n[5] = {n2, n3, n4, n5, n6};
                for (int i = 0; i < rank; ++i) {
                  starts = std::to_string(n[i]) + " " + starts;
                }
                std::string strides = "";
                for (int i = 0; i < rank; ++i) {
                  strides += " 1";
                }
                std::string counts = "";
                for (int i = 0; i < rank; ++i) {
                  counts += "1 ";
                }
                xdmf << StringPrintf(
                    "        <DataItem ItemType=\"HyperSlab\" Dimensions=\"%d\">\n"
                    "          <DataItem Dimensions=\"3 %d\" Format=\"XML\">\n"
                    "            %s1\n"
                    "            1%s\n"
                    "            %s%d\n"
                    "          </DataItem>\n"
                    "        </DataItem>\n",
                    particle_count, rank + 1, starts.c_str(), strides.c_str(),
                    counts.c_str(), particle_count);
                xdmf << ParticleDatasetRef("        ", swmname, varname, hdffile,
                                           varinfo.swtype, extradims.c_str(),
                                           particle_count);
              } else {
                xdmf << ParticleDatasetRef("        ", swmname, varname, hdffile,
                                           varinfo.swtype, "", particle_count);
              }
              xdmf << "      </Attribute>" << std::endl;
            }
          }
        }
      }
    }
  }
}
} // namespace impl
} // namespace XDMF
} // namespace parthenon

#endif // ENABLE_HDF5
