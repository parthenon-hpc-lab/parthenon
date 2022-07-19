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

#include "loop_advection.hpp"

#include <iostream>
#include <utility>
#include <vector>

//#include "parthenon_mpi.hpp"
//#include "parthenon_manager.hpp"

using parthenon::DriverStatus;
using parthenon::MeshBlock;
using parthenon::Metadata;
using parthenon::Packages_t;
using parthenon::ParameterInput;
using parthenon::Params;
using parthenon::Real;
using parthenon::StateDescriptor;
using parthenon::TaskID;
using parthenon::TaskList;
using parthenon::TaskStatus;
using parthenon::CartDir;

namespace loop_advection_example {

Packages_t ProcessPackages(std::unique_ptr<ParameterInput> &pin) {
  Packages_t packages;
  auto package = std::make_shared<StateDescriptor>("LoopAdvection");

  Params &params = package->AllParams();

  Metadata m;
  std::vector<int> vec_size({1});///Vector size of one

  //The magnetic field, defined at the faces,  which is evolved
  m = Metadata({Metadata::Face, Metadata::Independent, Metadata::FillGhost},
               vec_size);
  package->AddField("B_face", m);

  //The magnetic field, defined at cell centers,  which is interpolated from cell faces
  m = Metadata({Metadata::Cell, Metadata::Dependent, Metadata::FillGhost},
               vec_size);
  package->AddField("B_cell", m);

  //The electric field, defined at the cells centers,  which is derived from v x B
  m = Metadata({Metadata::Cell, Metadata::Derived, Metadata::FillGhost},
               vec_size);
  package->AddField("E_cell", m);

  //The electric field, defined at cell edges,  which is derived from v x B
  m = Metadata({Metadata::Edge, Metadata::Derived, Metadata::FillGhost},
               vec_size);
  package->AddField("E_edge", m);

  // The divergence of the face centered magnetic field at cell centers
  m = Metadata({Metadata::Cell, Metadata::Derived});
  package->AddField("div_Bf_cell", m);

  // The divergence of the cell centered magnetic field at cell centers
  m = Metadata({Metadata::Cell, Metadata::Derived});
  package->AddField("div_Bc_cell", m);

  // The divergence of the electric_field at nodes
  m = Metadata({Metadata::Node, Metadata::Derived});
  package->AddField("div_E_node", m);

  packages.Add(package);
  return packages;
}

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  using parthenon::MetadataFlag;

  auto &md = pmb->meshblock_data.Get();

  auto pkg = pmb->packages.Get("loop_advection_package");
  const auto &amp = pkg->Param<Real>("amp");
  const auto &R = pkg->Param<Real>("R");
  const auto &v1 = pkg->Param<Real>("v1");
  const auto &v2 = pkg->Param<Real>("v2");


  const auto &tup =  SparsePack<>::Make(pmd->meshblock_data.Get(),{"E_edge","B_face"});
  const auto &v = std::get<0>(tup);
  const auto &pack_map = std::get<1>(tup);

  const auto &A_vidx = pack_map["E_edge"];
  const auto &B_face_vidx = pack_map["B_face"];

  //Initialize vector potential at cell edges
  const auto &A = md->PackVars(std::vector<std::string>{"E_edge"});
  for( int edge_dir=1; edge_dir <= 3; edge_dir++){
    CartDir edge_dir_enum = static_cast<CartDir>(edge_dir);

    IndexRange ib = pmb->cellbounds.GetEdgeBoundsI(IndexDomain::entire,edge_dir_enum);
    IndexRange jb = pmb->cellbounds.GetEdgeBoundsJ(IndexDomain::entire,edge_dir_enum);
    IndexRange kb = pmb->cellbounds.GetEdgeBoundsK(IndexDomain::entire,edge_dir_enum);

    pmb->par_for(
      "LoopAdvection::ProblemGenerator::init_A.x"+(edge_dir)+"e", 0, vf.dim(5)-1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const auto &coords = E.GetCoords(b);
        const Real r = sqrt( x*x  + y*y + z*z );


        const Real x = (edge_dir == 1) ? coords.x1v(i) : coords.x1f(i);
        const Real y = (edge_dir == 2) ? coords.x2v(j) : coords.x2f(j);
        const Real z = (edge_dir == 3) ? coords.x3v(k) : coords.x2f(k);

        const Real ax = 0;
        const Real ay = 0;
        const Real az = max( amp*(R - r),0);

        v( b, edge_dir, A_vidx, k, j, i) = (edge_dir == 1 ) ? ax :
                                   (edge_dir == 2 ) ? ay : az;
      });
  }


  //Initialize magnetic field from curl of vector potential
  //Computing each direction individually (not necessarily the fastest method)
  for( int face_dir=1; face_dir <= 3; face_dir++){
    CartDir face_dir_enum = static_cast<CartDir>(face_dir);

    IndexRange ib = pmb->cellbounds.GetFaceBoundsI(IndexDomain::entire,face_dir_enum);
    IndexRange jb = pmb->cellbounds.GetFaceBoundsJ(IndexDomain::entire,face_dir_enum);
    IndexRange kb = pmb->cellbounds.GetFaceBoundsK(IndexDomain::entire,face_dir_enum);

    pmb->par_for(
      "LoopAdvection::ProblemGenerator::init_B.x"+(face_dir)+"f", 0, vf.dim(5)-1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const auto &coords = B.GetCoords(b);

        //Compute the Curl of E
        const int cart_dir_p1 = (cart_dir    )%3 + 1;
        const int cart_dir_p2 = (cart_dir + 1)%3 + 1;

        const Real dp1 = (cart_dir == 1) ? coords.dx1(i) :
                        +(cart_dir == 2) ? coords.dx2(j) :
                                           coords.dx3(k);
        const Real dp2 = (cart_dir == 1) ? coords.dx1(i) :
                        +(cart_dir == 2) ? coords.dx2(j) :
                                           coords.dx3(k);

        //First term: dEzdy, dExdz, or dEydx
        const Real dEp2dp1 =(v.cmpt( b, cart_dir_p2, A_vidx,  k, j, i)
                            -v.cmpt( b, cart_dir_p2, A_vidx,  k - (cart_dir_p1 == 3),
                                                j - (cart_dir_p1 == 2),
                                                i - (cart_dir_p1 == 1)))/dp1;

        //Second term: dEydz, dEzdx, or dExdy
        const Real dEp1dp2 =(v.cmpt( b, cart_dir_p1, A_vidx, k, j, i)
                            -v.cmpt( b, cart_dir_p1, A_vidx, k - (cart_dir_p2 == 3),
                                                j - (cart_dir_p2 == 2),
                                                i - (cart_dir_p2 == 1)))/dp2;

        v.cmpt( b, cart_dir, B_face_vidx, b, k, j, i) = -(dEp2dp1 - dEp1dp2);
      });
  }

}

// Compute v x B to compute E, then take the curl to compute dBdt
parthenon::TaskStatus calc_dBdt(parthenon::MeshBlock *pmb){

  auto pkg = pmb->packages.Get("loop_advection_package");

  const auto &v1 = pkg->Param<Real>("v1");
  const auto &v2 = pkg->Param<Real>("v2");
  const Real v3 = 0;

  const auto &tup =  SparsePack<>::Make(pmd->meshblock_data.Get(),{"B_face","B_cell","E_cell","E_edge"});
  const auto &v = std::get<0>(tup);
  const auto &pack_map = std::get<1>(tup);

  const auto &B_face_vidx = pack_map["B_face"];
  const auto &B_cell_vidx = pack_map["B_cell"];
  const auto &E_cell_vidx = pack_map["E_cell"];
  const auto &E_edge_vidx = pack_map["E_edge"];
  const auto &dBdt_face_vidx = B_face_vidx;

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  //First, compute the cell centered electric field 
  {

    pmb->par_for(
      "LoopAdvection::ProblemGenerator:calc_E_cell", 0, vf.dim(5)-1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {

        //Average B from the faces to the cell centers
        v(b, B_cell_vidx+0, k, j, i) = 0.5 * (v.cmpt( b, 1, B_face_vidx, k, j, i) + v.cmpt(b, 1, B_face_vidx,  k, j, i+1));
        v(b, B_cell_vidx+1, k, j, i) = 0.5 * (v.cmpt( b, 2, B_face_vidx, k, j, i) + v.cmpt(b, 2, B_face_vidx,  k, j+1, i));
        v(b, B_cell_vidx+2, k, j, i) = 0.5 * (v.cmpt( b, 3, B_face_vidx, k, j, i) + v.cmpt(b, 3, B_face_vidx,  k+1, j, i));

        //Compute E_cell = v X B
        v(b, E_cell_vidx+0, k, j, i) = v2 * B3 - v3 * B2;
        v(b, E_cell_vidx+1, k, j, i) = v3 * B1 - v1 * B3;
        v(b, E_cell_vidx+2, k, j, i) = v1 * B2 - v2 * B1;

      });
  }

  //Then interpolate E_cell to E_edge
  //(Do it in one loop for demonstrative purposes)
  {

    pmb->par_for(
      "LoopAdvection::ProblemGenerator:calc_E_edge", 0, vf.dim(5)-1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        //Average E from the cell centers to cell edges
        if( k+1 < kb.e && j+1 < jb.e) 
          v.cmpt(b, 1, E_edge_idx, k, j, i) = v.cmpt(b, 1, E_face_vidx,   k,   j,   i) + v.cmpt(b, 1, E_face_vidx,   k, j+1,   i);
                                            + v.cmpt(b, 1, E_face_vidx, k+1,   j,   i) + v.cmpt(b, 1, E_face_vidx, k+1, j+1,   i);
        if( j+1 < jb.e && i+1 < ib.e) 
          v.cmpt(b, 2, E_edge_idx, k, j, i) = v.cmpt(b, 2, E_face_vidx,   k,   j,   i) + v.cmpt(b, 2, E_face_vidx,   k,   j, i+1);
                                            + v.cmpt(b, 2, E_face_vidx, k+1,   j,   i) + v.cmpt(b, 2, E_face_vidx, k+1,   j, i+1);
        if( j+1 < jb.e && k+1 < kb.e) 
          v.cmpt(b, 3, E_edge_idx, k, j, i) = v.cmpt(b, 3, E_face_vidx,   k,   j,   i) + v.cmpt(b, 3, E_face_vidx,   k,   j, i+1);
                                            + v.cmpt(b, 3, E_face_vidx,   k, j+1,   i) + v.cmpt(b, 3, E_face_vidx,   k, j+1, i+1);



        });
    }

  //After averaging from faces to edges, reduce bounds by 1
  //FIXME(forrestglines): Is this how to shrink bounds?
  ib.e -= 1;
  jb.e -= 1;
  kb.e -= 1;

  //Last, take the curl of E_edge to cell faces to compute dBdt
  //(Do it in one loop for demonstrative purposes)
  {

    pmb->par_for(
      "LoopAdvection::ProblemGenerator:curl_E_edge", 0, vf.dim(5)-1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const auto &coords = v.GetCoordinates(b);

        if( k+1 < kb.e && j+1 < jb.e) 
          v.cmpt(b, 1, dBdt_face_vidx, k, j, i) = (v.cmpt(b, 3, E_edge_vidx k, j+1, i) - v.cmpt(b, 3, E_edge_vidx k, j, i))/coords.dx2(j)
                                                - (v.cmpt(b, 2, E_edge_vidx k+1, j, i) - v.cmpt(b, 2, E_edge_vidx k, j, i))/coords.dx3(k);
        if( j+1 < jb.e && i+1 < ib.e) 
          v.cmpt(b, 2, dBdt_face_vidx, k, j, i) = (v.cmpt(b, 1, E_edge_vidx k+1, j, i) - v.cmpt(b, 1, E_edge_vidx k, j, i))/coords.dx3(k)
                                                - (v.cmpt(b, 3, E_edge_vidx k, j, i+1) - v.cmpt(b, 3, E_edge_vidx k, j, i))/coords.dx1(i);
        if( j+1 < jb.e && k+1 < kb.e) 
          v.cmpt(b, 3, dBdt_face_vidx, k, j, i) = (v.cmpt(b, 2, E_edge_vidx k, j, i+1) - v.cmpt(b, 2, E_edge_vidx k, j, i))/coords.dx3(k)
                                                - (v.cmpt(b, 1, E_edge_vidx k, j+1, i) - v.cmpt(b, 1, E_edge_vidx k, j, i))/coords.dx2(j);
      });
  }

  //After averaging from faces to edges, reduce bounds by 1 - so dBdt doesn't include ghosts
  //FIXME(forrestglines): How to shrink bounds?
}

//========================================================================================
//! \fn void Mesh::UserWorkAfterLoop(ParameterInput *pin, SimTime &tm)
//  \brief Compute L1 error in advection test and output to file
//========================================================================================

void UserWorkAfterLoop(Mesh *mesh, ParameterInput *pin, SimTime &tm) {
  // Compute divergence of electric field at nodes
  {
    IndexRange ib = pmb->cellbounds.GetNodeBoundsI(IndexDomain::interior);
    IndexRange jb = pmb->cellbounds.GetNodeBoundsJ(IndexDomain::interior);
    IndexRange kb = pmb->cellbounds.GetNodeBoundsK(IndexDomain::interior);

    const auto &tup =  SparsePack<>::Make(pmd->meshblock_data.Get(),{"E_edge","div_E_node"});
    const auto &v = std::get<0>(tup);
    const auto &pack_map = std::get<1>(tup);

    const auto &E_edge_vidx = pack_map["E_edge"];
    const auto &div_E_node_vidx = pack_map["div_E_node"];

    // Compute the div of E at nodes(should be zero?)
    pmb->par_for(
      "LoopAdvection::calc_div_E", 0, vf.dim(5)-1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const auto &coords = v.GetCoordinates(b);

        v(b, div_E_node_vidx, k, j, i) = (v.cmpt(b, 1, E_edge_vidx, k, j, i) -
                                          v.cmpt(b, 1, E_edge_vidx, k, j, i - 1)) /
                                             coords.dx1(i) +
                                         (v.cmpt(b, 2, E_edge_vidx, k, j, i) -
                                          v.cmpt(b, 2, E_edge_vidx, k, j - 1, i)) /
                                             coords.dx2(j) +
                                         (v.cmpt(b, 3, E_edge_vidx, k, j, i) -
                                          v.cmpt(b, 3, E_edge_vidx, k - 1, j, i)) /
                                             coords.dx3(k);

      });
  }

  // Compute divergence of magnetic field at nodes
  {
    IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
    IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
    IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

    const auto &B_face = pmd->meshblock_data.Get()->PackVars(
        std::vector<std::string>({"B_face"}));
    const auto &div_Bf_cell = pmd->meshblock_data.Get()->PackVars(
        std::vector<std::string>({"div_Bf_cell"}));

    const auto &B_cell = pmd->meshblock_data.Get()->PackVars(
        std::vector<std::string>({"B_cell"}));
    const auto &div_Bc_cell = pmd->meshblock_data.Get()->PackVars(
        std::vector<std::string>({"div_Bc_cell"}));

    const auto &tup = SparsePack<>::Make(
        pmd->meshblock_data.Get(), {"B_face", "div_Bf_cell", "B_cell", "div_Bc_cell"});
    const auto &v = std::get<0>(tup);
    const auto &pack_map = std::get<1>(tup);

    const auto &B_face_vidx = pack_map["B_face"];
    const auto &div_Bf_cell_vidx = pack_map["div_Bf_cell"];
    const auto &B_cell_vidx = pack_map["B_cell"];
    const auto &div_Bc_cell_vidx = pack_map["div_Bc_cell"];

    // Compute the div of B at nodes(should be zero)
    pmb->par_for(
      "LoopAdvection::calc_div_B", 0, vf.dim(5)-1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const auto &coords = v.GetCoordinates(b);

        v( b, div_Bf_cell_vidx, k, j, i) = (v.cmpt( b, 1, B_face_vidx, k, j, i) - v.cmpt( b, 1, B_face_vidx, k, j, i-1))/coords.dx1(i)
                                          +(v.cmpt( b, 2, B_face_vidx, k, j, i) - v.cmpt( b, 2, B_face_vidx, k, j-1, i))/coords.dx2(j)
                                          +(v.cmpt( b, 3, B_face_vidx, k, j, i) - v.cmpt( b, 3, B_face_vidx, k-1, j, i))/coords.dx3(k);
        v( b, div_Bc_cell_vidx, k, j, i) = (v( b, B_cell_vidx+0, k, j, i+1) - v( b, B_cell_vidx+0, k, j, i-1))/(2*coords.dx1(i))
                                          +(v( b, B_cell_vidx+1, k, j+1, i) - v( b, B_cell_vidx+1, k, j-1, i))/(2*coords.dx2(j))
                                          +(v( b, B_cell_vidx+1, k+1, j, i) - v( b, B_cell_vidx+2, k-1, j, i))/(2*coords.dx3(k));

      });
  }
}

} // namespace LoopAdvection
