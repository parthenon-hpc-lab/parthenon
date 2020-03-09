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

#define CATCH_CONFIG_RUNNER
#include <fstream>

#include <catch2/catch.hpp> 

#include "mesh/mesh.hpp"

int main(int argc, char * argv[]){

  Catch::Session session;

  int provided;
  MPI_Init_thread(&argc,&argv,MPI_THREAD_MULTIPLE, &provided);
  //const int result = Catch::Session().run(argc,argv);
  session.run();

  MPI_Finalize();

  return 0;
}

TEST_CASE("Testing Mesh Constructor"){

  /// Used to set global rank
  MPI_Comm_size(MPI_COMM_WORLD, &(Globals::nranks));

  int rank;
  MPI_Comm_rank( MPI_COMM_WORLD, &rank);

  std::stringstream str_stream;
  str_stream << "<time>" << std::endl;
  str_stream << "nlim        = -1        # cycle limit" << std::endl;
  str_stream << "tlim        = 0.25      # time limit" << std::endl;
  str_stream << "ncycle_out  = 25        # interval for stdout summary info" << std::endl;
  str_stream << "" << std::endl;
  str_stream << "<mesh>" << std::endl;
  str_stream << "nx1         = 16       # Number of zones in X1-direction" << std::endl;
  str_stream << "x1min       = -0.5      # minimum value of X1" << std::endl;
  str_stream << "x1max       = 0.5       # maximum value of X1" << std::endl;
  str_stream << "ix1_bc      = outstr_streamow   # Inner-X1 boundary condition str_streamag" << std::endl;
  str_stream << "ox1_bc      = outstr_streamow   # Outer-X1 boundary condition str_streamag" << std::endl;
  str_stream << "" << std::endl;
  str_stream << "nx2         = 24         # Number of zones in X2-direction" << std::endl;
  str_stream << "x2min       = -0.6      # minimum value of X2" << std::endl;
  str_stream << "x2max       = 0.6       # maximum value of X2" << std::endl;
  str_stream << "ix2_bc      = periodic  # Inner-X2 boundary condition str_streamag" << std::endl;
  str_stream << "ox2_bc      = periodic  # Outer-X2 boundary condition str_streamag" << std::endl;
  str_stream << "" << std::endl;
  str_stream << "nx3         = 12         # Number of zones in X3-direction" << std::endl;
  str_stream << "x3min       = -0.7      # minimum value of X3" << std::endl;
  str_stream << "x3max       = 0.7       # maximum value of X3" << std::endl;
  str_stream << "ix3_bc      = periodic  # Inner-X3 boundary condition str_streamag" << std::endl;
  str_stream << "ox3_bc      = periodic  # Outer-X3 boundary condition str_streamfgag" << std::endl;
  str_stream << "" << std::endl;
  str_stream << "num_threads = 2         # maximum number of OMP threads" << std::endl;
  str_stream << "" << std::endl;
  str_stream << "<meshblock>" << std::endl;
  str_stream << "nx1 = 8" << std::endl;
  str_stream << "nx2 = 4" << std::endl;
  str_stream << "nx3 = 6" << std::endl;

  ParameterInput pin;
  pin.LoadFromStream(str_stream);

  std::vector<std::shared_ptr<MaterialPropertiesInterface>> materials;
  std::map<std::string, std::shared_ptr<StateDescriptor>> physics;
  PreFillDerivedFunc pre_fill_derived;
  int test_flag=0;

  Mesh mesh(
      &pin,
      materials,
      physics,
      pre_fill_derived,
      test_flag);

  /// Should be == 2, equivalent to the number of threads in the input file
  REQUIRE( mesh.GetNumMeshThreads() == 2 ); 
  /// The total number of mesh blocks per rank is found by first finding the total number of mesh 
  /// blocks by dividing the total number of cells which
  /// in this case is nx1 x nx2 x nx3 = 16 x 24 x 12 = 4608 by the total number of cells in a block
  /// which is under the <meshblock> tag as nx1 x nx2 x nx3 = 8 x 4 x 6 = 192
  ///
  /// 4608 / 192 = 24 mesh blocks
  /// 
  /// The total number of ranks is nranks so if run with:
  ///
  /// mpirun -n 4 test_integration_mesh 
  ///
  /// should be 4
  ///
  /// 24 / 4 = 6 mesh blocks per rank
  ///
  int total_num_cells = mesh.mesh_size.nx1 * mesh.mesh_size.nx2 * mesh.mesh_size.nx3;

  int mesh_x1_cells = stoi(pin.GetOrAddString("meshblock","nx1","none"));
  int mesh_x2_cells = stoi(pin.GetOrAddString("meshblock","nx2","none"));
  int mesh_x3_cells = stoi(pin.GetOrAddString("meshblock","nx3","none"));
  int total_cells_per_block = mesh_x1_cells*mesh_x2_cells*mesh_x3_cells; 

  int total_num_mesh_blocks = total_num_cells/total_cells_per_block;
  int mesh_blocks_per_rank = total_num_mesh_blocks/Globals::nranks; 

  REQUIRE( mesh.GetNumMeshBlocksThisRank(0) == mesh_blocks_per_rank );
  /// Total number of blocks 
  REQUIRE( mesh.nbtotal == total_num_mesh_blocks );
  /// number of deleted blocks
  REQUIRE( mesh.nbnew == 0 );
  /// number of new blocks
  REQUIRE( mesh.nbdel == 0 );
  /// Steps since load balancing
  REQUIRE( mesh.step_since_lb == 0 );

  REQUIRE( mesh.mesh_size.x1min == Approx( -0.5 ) ); 
  REQUIRE( mesh.mesh_size.x2min == Approx( -0.6 ) ); 
  REQUIRE( mesh.mesh_size.x3min == Approx( -0.7 ) ); 

  REQUIRE( mesh.mesh_size.x1max == Approx( 0.5 ) ); 
  REQUIRE( mesh.mesh_size.x2max == Approx( 0.6 ) ); 
  REQUIRE( mesh.mesh_size.x3max == Approx( 0.7 ) ); 

  REQUIRE( mesh.mesh_size.nx1 == 16 );
  REQUIRE( mesh.mesh_size.nx2 == 24 );
  REQUIRE( mesh.mesh_size.nx3 == 12 );

  /// Outflow x2 in and x1 out
  REQUIRE( mesh.mesh_bcs[0] == BoundaryFlag::outflow  );
  REQUIRE( mesh.mesh_bcs[1] == BoundaryFlag::outflow );
  /// Periodic for x2 and x3
  REQUIRE( mesh.mesh_bcs[2] == BoundaryFlag::periodic);
  REQUIRE( mesh.mesh_bcs[3] == BoundaryFlag::periodic);
  REQUIRE( mesh.mesh_bcs[4] == BoundaryFlag::periodic);
  REQUIRE( mesh.mesh_bcs[5] == BoundaryFlag::periodic);

  /// Flags indicating the dimensions, because we are dealing with a 3d problem both flags should be
  /// true
  REQUIRE( mesh.f2 == true );
  REQUIRE( mesh.f3 == true );

  REQUIRE( mesh.adaptive == false);
  REQUIRE( mesh.multilevel == false);
  REQUIRE( mesh.ndim == 3);

  /// Check that time output was stored correctly
  REQUIRE( mesh.ncycle_out == 25 );
  REQUIRE( mesh.ncycle == 0 );
  REQUIRE( mesh.nlim == -1 );
  REQUIRE( mesh.tlim == Approx(0.25));
  REQUIRE( mesh.dt_diagnostics == -1);
}


