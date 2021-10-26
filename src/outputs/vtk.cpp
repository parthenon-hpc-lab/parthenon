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
//! \file vtk.cpp
//  \brief writes output data in (legacy) vtk format.
//  Data is written in RECTILINEAR_GRID geometry, in BINARY format, and in FLOAT type
//  Writes one file per MeshBlock.

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "coordinates/coordinates.hpp"
#include "defs.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "outputs/outputs.hpp"
#include "parthenon_arrays.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {

//----------------------------------------------------------------------------------------
// Functions to detect big endian machine, and to byte-swap 32-bit words.  The vtk
// legacy format requires data to be stored as big-endian.

int IsBigEndian() {
  std::int32_t n = 1;
  // careful! although int -> char * -> int round-trip conversion is safe,
  // an arbitrary char* may not be converted to int*
  char *ep = reinterpret_cast<char *>(&n);
  return (*ep == 0); // Returns 1 (true) on a big endian machine
}

// namespace {
// inline void Swap4Bytes(void *vdat) {
//   char tmp, *dat = static_cast<char *>(vdat);
//   tmp = dat[0];
//   dat[0] = dat[3];
//   dat[3] = tmp;
//   tmp = dat[1];
//   dat[1] = dat[2];
//   dat[2] = tmp;
// }
// } // namespace

//----------------------------------------------------------------------------------------
//! \fn void VTKOutput:::WriteOutputFile(Mesh *pm)
//  \brief Cycles over all MeshBlocks and writes OutputData in (legacy) vtk format, one
//         MeshBlock per file

void VTKOutput::WriteContainer(SimTime &tm, Mesh *pm, ParameterInput *pin, bool flag) {
  throw std::runtime_error(std::string(__func__) + " is not implemented");
  /*
  int big_end = IsBigEndian(); // =1 on big endian machine

  // Loop over MeshBlocks
  for (auto &pmb : pm->block_list) {
    // set start/end array indices depending on whether ghost zones are included
    IndexDomain domain = IndexDomain::interior;
    if (output_params.include_ghost_zones) {
      domain = IndexDomain::entire;
    }
    // build doubly linked list of OutputData nodes (setting data ptrs to appropriate
    // quantity on MeshBlock for each node), then slice/sum as needed
    // create filename: "file_basename"+ "."+"blockid"+"."+"file_id"+"."+XXXXX+".vtk",
    // where XXXXX = 5-digit file_number
    std::string fname;
    char number[6];
    std::snprintf(number, sizeof(number), "%05d", output_params.file_number);
    char blockid[12];
    std::snprintf(blockid, sizeof(blockid), "block%d", pmb->gid);

    fname.assign(output_params.file_basename);
    fname.append(".N.");
    fname.append(blockid);
    fname.append(".");
    fname.append(output_params.file_id);
    fname.append(".");
    fname.append(number);
    fname.append(".vtk");

    // open file for output
    FILE *pfile;
    std::stringstream msg;
    if ((pfile = std::fopen(fname.c_str(), "w")) == nullptr) {
      msg << "### FATAL ERROR in function [VTKOutput::WriteOutputFile]" << std::endl
          << "Output file '" << fname << "' could not be opened" << std::endl;
      PARTHENON_FAIL(msg);
    }

    // There are five basic parts to the VTK "legacy" file format.
    //  1. Write file version and identifier
    std::fprintf(pfile, "# vtk DataFile Version 2.0\n");

    //  2. Header
    std::fprintf(pfile, "# Athena++ data at time=%e", tm.time);
    std::fprintf(pfile, "  cycle=%d", tm.ncycle);
    std::fprintf(pfile, "  variables=%s \n", output_params.variable.c_str());

    //  3. File format
    std::fprintf(pfile, "BINARY\n");

    //  4. Dataset structure
    int ncells1 = pmb->cellbounds.ncellsi(domain);
    int ncells2 = pmb->cellbounds.ncellsj(domain);
    int ncells3 = pmb->cellbounds.ncellsk(domain);
    int ncoord1 = ncells1;
    if (ncells1 > 1) ncoord1++;
    int ncoord2 = ncells2;
    if (ncells2 > 1) ncoord2++;
    int ncoord3 = ncells3;
    if (ncells3 > 1) ncoord3++;

    float *data;
    int ndata = std::max(ncoord1, ncoord2);
    ndata = std::max(ndata, ncoord3);
    data = new float[3 * ndata + 1];

    // Specify the type of data, dimensions, and coordinates.  If N>1, then write N+1
    // cell faces as binary floats.  If N=1, then write 1 cell center position.
    std::fprintf(pfile, "DATASET RECTILINEAR_GRID\n");
    std::fprintf(pfile, "DIMENSIONS %d %d %d\n", ncoord1, ncoord2, ncoord3);

    IndexRange out_ib = pmb->cellbounds.GetBoundsI(domain);
    IndexRange out_jb = pmb->cellbounds.GetBoundsJ(domain);
    IndexRange out_kb = pmb->cellbounds.GetBoundsK(domain);
    // write x1-coordinates as binary float in big endian order
    std::fprintf(pfile, "X_COORDINATES %d float\n", ncoord1);
    if (ncells1 == 1) {
      data[0] = static_cast<float>(pmb->coords.x1v(out_ib.s));
    } else {
      for (int i = out_ib.s; i <= out_ib.e + 1; ++i) {
        data[i - out_ib.s] = static_cast<float>(pmb->coords.x1f(i));
      }
    }
    if (!big_end) {
      for (int i = 0; i < ncoord1; ++i)
        Swap4Bytes(&data[i]);
    }
    std::fwrite(data, sizeof(float), static_cast<std::size_t>(ncoord1), pfile);

    // write x2-coordinates as binary float in big endian order
    std::fprintf(pfile, "\nY_COORDINATES %d float\n", ncoord2);
    if (ncells2 == 1) {
      data[0] = static_cast<float>(pmb->coords.x2v(out_jb.s));
    } else {
      for (int j = out_jb.s; j <= out_jb.e + 1; ++j) {
        data[j - out_jb.s] = static_cast<float>(pmb->coords.x2f(j));
      }
    }
    if (!big_end) {
      for (int i = 0; i < ncoord2; ++i)
        Swap4Bytes(&data[i]);
    }
    std::fwrite(data, sizeof(float), static_cast<std::size_t>(ncoord2), pfile);

    // write x3-coordinates as binary float in big endian order
    std::fprintf(pfile, "\nZ_COORDINATES %d float\n", ncoord3);
    if (ncells3 == 1) {
      data[0] = static_cast<float>(pmb->coords.x3v(out_kb.s));
    } else {
      for (int k = out_kb.s; k <= out_kb.e + 1; ++k) {
        data[k - out_kb.s] = static_cast<float>(pmb->coords.x3f(k));
      }
    }
    if (!big_end) {
      for (int i = 0; i < ncoord3; ++i)
        Swap4Bytes(&data[i]);
    }
    std::fwrite(data, sizeof(float), static_cast<std::size_t>(ncoord3), pfile);

    //  5. Data.  An arbitrary number of scalars and vectors can be written (every node
    //  in the OutputData doubly linked lists), all in binary floats format

    std::fprintf(pfile, "\nCELL_DATA %d", pmb->cellbounds.GetTotal(domain));
    // reset container iterator to point to current block data
    auto ci =
        MeshBlockDataIterator<Real>(pmb->meshblock_data.Get(), {Metadata::Graphics});
    for (auto &v : ci.vars) {
      if (!data) {
        std::cout << "____________________SKIPPPING:" << v->label() << std::endl;
        continue;
      }
      std::fprintf(pfile, "\nLOOKUP_TABLE default\n");
      for (int k = out_kb.s; k <= out_kb.e; k++) {
        for (int j = out_jb.s; j <= out_jb.e; j++) {
          int index = 0;
          for (int i = out_ib.s; i <= out_ib.e; i++, index++) {
            data[(i - out_ib.s) + index] = (*v)(k, j, i);
          }

          // write data in big endian order
          if (!big_end) {
            for (int i = 0; i < (ncells1); ++i)
              Swap4Bytes(&data[i]);
          }
          std::fwrite(data, sizeof(float), static_cast<std::size_t>(ncells1), pfile);
        }
      }
    }

    // don't forget to close the output file and clean up ptrs to data in OutputData
    std::fclose(pfile);
    delete[] data;
  } // end loop over MeshBlocks

  // increment counters
  output_params.file_number++;
  output_params.next_time += output_params.dt;
  pin->SetInteger(output_params.block_name, "file_number", output_params.file_number);
  pin->SetReal(output_params.block_name, "next_time", output_params.next_time);

  return;
  */
}
void VTKOutput::WriteOutputFile(Mesh *pm, ParameterInput *pin, SimTime *tm,
                                const SignalHandler::OutputSignal signal) {
  throw std::runtime_error(std::string(__func__) + " is not implemented");
}

} // namespace parthenon
