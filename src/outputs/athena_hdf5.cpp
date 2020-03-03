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

#if 0
// New c interface being tested

// C headers
#include <defs.hpp>

// C++ headers
#include <fstream>    // ofstream, quoted
#include <iomanip>

// Athena++ headers
#include "athena.hpp"
#include "athena_arrays.hpp"
#include "coordinates/coordinates.hpp"
#include "globals.hpp"
#include "interface/ContainerIterator.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "outputs.hpp"

// Only proceed if HDF5 output enabled
#ifdef HDF5OUTPUT

// External library headers
#include "H5Cpp.h"
using namespace H5;


// macro to write an attribute to a HDF file
#define WRITE_H5A(name, pData, file, dSpace, dSet, myAttrPredType, myDataPredType) { \
    Attribute attribute = dSet.createAttribute(name, myAttrPredType, dSpace);        \
    attribute.write(myDataPredType, pData);				             \
  }
#define PREDINT32 PredType::NATIVE_INT32
#define PREDFLOAT64 PredType::NATIVE_DOUBLE

// XDMF subroutine to write a dataitem that refers to an HDF array
static void writeXdmfArrayRef(std::ofstream& fid, const std::string& prefix,
			      const std::string& hdfPath, const std::string& label,
			      const hsize_t* dims, const int& ndims,
			      const std::string& theType, const int& precision) {
  fid << prefix << R"(<DataItem Format="HDF" Dimensions=")" << dims[0];
  for (int i=1; i<ndims; i++) fid << " " << dims[i];
  fid << "\"";
  fid << " Name=" << std::quoted(label)
      << " NumberType=" << std::quoted(theType)
      << R"( Precision=")" << precision << R"(")"
      << " >" << std::endl;
  fid << prefix +"  " << hdfPath << label << "</DataItem>" << std::endl;
}

// XDMF subroutine to write a variable that reads from a HDF file
static void writeXdmfVariableRef(std::ofstream& fid, const std::string& prefix,
				 const std::string& hdfPath, const std::string& label,
				 const hsize_t* dims, const int& ndims,
				 const std::string& theType, const int& precision) {
  fid << prefix << "<Attribute Name=" << std::quoted(label) << R"( Center="Cell">)" << std::endl;
  writeXdmfArrayRef(fid, prefix+"  ", hdfPath, label, dims, ndims, theType, precision);
  fid << prefix << "</Attribute>" << std::endl;
}


//----------------------------------------------------------------------------------------
//! \fn void ATHDF5Output:::WriteOutputFile(Mesh *pm, ParameterInput *pin, bool flag)
//  \brief Cycles over all MeshBlocks and writes OutputData in the Athena++ HDF5 format,
//         one file per output using parallel IO.
void ATHDF5Output::WriteOutputFile(Mesh *pm, ParameterInput *pin, bool flag) {

  // writes all graphics variables to hdf file
  // HDF5 structures
  // Also writes companion xdmf file
  MeshBlock *pmb = pm->pblock;
  int max_blocks_global = pm->nbtotal;
  int max_blocks_local = pm->nblist[Globals::my_rank];
  int num_blocks_local = 0;

  // shooting a blank just for getting the variable names
  out_is = pmb->is; out_ie = pmb->ie;
  out_js = pmb->js; out_je = pmb->je;
  out_ks = pmb->ks; out_ke = pmb->ke;
  if (output_params.include_ghost_zones) {
    out_is -= NGHOST; out_ie += NGHOST;
    if (out_js != out_je) {out_js -= NGHOST; out_je += NGHOST;}
    if (out_ks != out_ke) {out_ks -= NGHOST; out_ke += NGHOST;}
  }
  while(pmb != nullptr) {
    num_blocks_local++;
    pmb = pmb->next;
  }
  pmb = pm->pblock;
  // set output size
  nx1 = pmb->block_size.nx1;
  nx2 = pmb->block_size.nx2;
  nx3 = pmb->block_size.nx3;
  if (output_params.include_ghost_zones) {
    nx1 += 2*NGHOST;
    if (nx2 > 1) nx2 += 2*NGHOST;
    if (nx3 > 1) nx3 += 2*NGHOST;
  }

  // create dataspaces and types
  Attribute attribute;
  int dims_count[1] = {3};

  // open HDF5 file
  // Define output filename
  filename = std::string(output_params.file_basename);
  filename.append(".");
  filename.append(output_params.file_id);
  filename.append(".");
  std::stringstream file_number;
  file_number << std::setw(5) << std::setfill('0') << output_params.file_number;
  filename.append(file_number.str());
  filename.append(".hdf5");

  H5::H5File file(filename, H5F_ACC_TRUNC);

  std::string filename_aux(filename+std::string(".xdmf"));
  std::ofstream xdmf(filename_aux.c_str());

  // Write header
  xdmf << R"(<?xml version="1.0" ?>)" << std::endl;
  xdmf << R"(<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd">)" << std::endl;
  xdmf << R"(<Xdmf Version="3.0">)" << std::endl;
  xdmf << "  <Domain>" << std::endl;

  // write number of cycles
  hsize_t dims[4]={1,0,0,0};
  dims[0] = 1;
  DataSpace myDSpace= DataSpace(1,dims);
  DataSet myDSet = file.createDataSet("Timestep",PREDINT32,myDSpace);
  std::string prefix="   ";

  // write number of cycles
  WRITE_H5A("/Timestep/NCycle", &pm->ncycle, file, myDSpace, myDSet, PREDINT32, PREDINT32);

  // Create the Blocks group and write simulation time
  Group gBlocks(file.createGroup("/Blocks"));
  WRITE_H5A("/Timestep/Time", &pm->time, file, myDSpace, myDSet, PREDFLOAT64, PREDFLOAT64);

  // xdmf: Create blocks and write simulation time
  xdmf << R"(    <Grid Name="Blocks" CollectionType="Spatial" GridType="Collection">)" << std::endl;
  xdmf << R"(      <Time Value=")" << pm->time << R"("/>)" << std::endl;
  xdmf << R"(      <Information Name="Cycle" Value=")" << pm->ncycle << R"("/>)" << std::endl;

  // write number of dimensions
  WRITE_H5A("/Timestep/NumDims", &pm->ndim, file, myDSpace, myDSet, PREDINT32, PREDINT32);

  // write number of mesh blocks
  WRITE_H5A("/Timestep/NumMeshBlocks", &num_blocks_local, file, myDSpace, myDSet, PREDINT32, PREDINT32);


  std::string hdfFile = filename;

  // write MeshBlock size
  int meshblock_size[3] = {nx1, nx2, nx3};
  const hsize_t xDims[1] = {3};
  myDSpace = DataSpace(1,xDims);
  myDSet = gBlocks.createDataSet("MeshBlockSize",PREDINT32,myDSpace);
  myDSet.write(meshblock_size, PREDINT32);

  // Write maximum refinement level
  int max_level = pm->current_level - pm->root_level;
  WRITE_H5A("/Timestep/MaxLevel", &max_level, file, myDSpace, myDSet, PREDINT32, PREDINT32);

  // Now write blocks
  pmb = pm->pblock;
  int idx=0;
  while (pmb != nullptr) {
    // create group for block
    DataSet dSet;
    DataSpace dSpace;
    Real *theData;
    Real Origin_ZYX[3] = {0.,0.,0.};
    Real dZdYdX[3] = {1.,1.,1.};

    // create group for this block
    char tmpName[64];
    snprintf(tmpName, 63, "/Blocks/%06d", idx);
    Group gMyBlock(file.createGroup(tmpName));
    int ilen = strlen(tmpName);

    // write ID
    dims[0] = 1;
    dSpace = DataSpace(1,dims);
    dSet = gMyBlock.createDataSet("ID",PREDINT32,dSpace);
    dSet.write(&idx, PREDINT32);

    // compute origin and size of mesh to be written
    theData = pmb->pcoord->x3f.data() + out_ks;
    Origin_ZYX[0] = theData[0];
    dZdYdX[0] = (theData[nx3] - theData[0])/(Real)nx3;

    theData = pmb->pcoord->x2f.data() + out_js;
    Origin_ZYX[1] = theData[0];
    dZdYdX[1] = (theData[nx2] - theData[0])/(Real)nx2;

    theData = pmb->pcoord->x1f.data() + out_is;
    Origin_ZYX[2] = theData[0];
    dZdYdX[2] = (theData[nx1] - theData[0])/(Real)nx1;

    // write origin to the HDF file
    dims[0] = 3;
    dSpace = DataSpace(1,dims);
    dSet = gMyBlock.createDataSet("Origin_ZYX",PREDFLOAT64,dSpace);
    dSet.write(Origin_ZYX, PREDFLOAT64);

    // write dZdYdX to the HDF file
    dims[0] = 3;
    dSpace = DataSpace(1,dims);
    dSet = gMyBlock.createDataSet("dZdYdX",PREDFLOAT64,dSpace);
    dSet.write(dZdYdX, PREDFLOAT64);

    // write grid to XDMF file
    xdmf << "      <Grid Name=" << std::quoted(tmpName) << R"( GridType="Uniform">)" << std::endl;
    xdmf << R"(        <Geometry GeometryType="ORIGIN_DXDYDZ">)" << std::endl;
    writeXdmfArrayRef(xdmf, "          ", hdfFile+":"+tmpName+"/","Origin_ZYX", dims, 1, "Float", 8);
    writeXdmfArrayRef(xdmf, "          ", hdfFile+":"+tmpName+"/","dZdYdX", dims, 1, "Float", 8);
    xdmf << "        </Geometry>" << std::endl;
    xdmf << R"(        <Topology Dimensions=")"
	 << nx3+1
	 << " " << nx2+1
	 << " " << nx1+1
	 << R"(" Type="3DCoRectMesh"/>)" << std::endl;

    // write graphics variables
    size_t baseSize = nx1*nx2*nx3;
    dims[0] = nx1;
    dims[1] = nx2;
    dims[2] = nx3;
    int ndims = 1;
    if ( nx3 > 1) { ndims = 3; }
    else if (nx2 > 1) { ndims = 2; }

    auto ci = ContainerIterator<Real>(pmb->real_container,{Metadata::graphics});

    int maxV = 1;
    for (auto &v : ci.vars) {
      const size_t vlen = v->GetDim4();
      maxV = (maxV<vlen?vlen:maxV);
    }
    Real *tmpData = new Real[baseSize*maxV];
    for (auto &v : ci.vars) {
      const size_t vlen = v->GetDim4();
      dims[ndims] = vlen;
      if ( vlen > 1 ) ndims += 1;
      int index;
      index=0;
      for (int k = out_ks; k <= out_ke; k++) {
        for (int j = out_js; j <= out_je; j++) {
          for (int i = out_is; i <= out_ie; i++) {
            for (int l = 0; l < vlen; l++, index++) {
              tmpData[index] = (*v)(l,k,j,i);
            }
          }
        }
      }
      dSpace = DataSpace(ndims,dims);
      dSet = gMyBlock.createDataSet(v->label(),PREDFLOAT64,dSpace);
      dSet.write(tmpData, PREDFLOAT64);
      writeXdmfVariableRef(xdmf, "        ", hdfFile+":"+tmpName+"/",v->label(), dims, ndims, "Float", 8);
      if ( vlen > 1 ) ndims -= 1;
    }

    xdmf << "      </Grid>" << std::endl;
    delete [] tmpData;
    gMyBlock.close();
    pmb = pmb->next;
    idx++;
  }

  // close blocks and HDF file
  gBlocks.close();
  file.close();

  // close xdmf headers and close file
  xdmf << "    </Grid>" << std::endl;
  xdmf << "  </Domain>" << std::endl;
  xdmf << "</Xdmf>" << std::endl;
  xdmf.close();

  // advance output parameters
  output_params.file_number++;
  output_params.next_time += output_params.dt;
  pin->SetInteger(output_params.block_name, "file_number", output_params.file_number);
  pin->SetReal(output_params.block_name, "next_time", output_params.next_time);
  return;
}


#endif  // HDF5OUTPUT
#endif // new c interface
