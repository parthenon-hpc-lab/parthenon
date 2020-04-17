
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

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <catch2/catch.hpp>

#include "basic_types.hpp"
#include "interface/container.hpp"
#include "interface/container_iterator.hpp"
#include "interface/metadata.hpp"
#include "interface/variable.hpp"
#include "kokkos_abstraction.hpp"
#include "parthenon_arrays.hpp"

using parthenon::CellVariableVector;
using parthenon::Container;
using parthenon::ContainerIterator;
using parthenon::DevSpace;
using parthenon::loop_pattern_mdrange_tag;
using parthenon::Metadata;
using parthenon::par_for;
using parthenon::ParArrayND;
using parthenon::ParArray4D;
using parthenon::Real;

TEST_CASE("Can pull variables from containers based on Metadata", "[ContainerIterator]") {
  GIVEN("A Container with a set of variables") {
    Container<Real> rc;
    Metadata m_in({Metadata::Independent});
    Metadata m_out;
    std::vector<int> scalar_block_size{16, 16, 16};
    std::vector<int> vector_block_size{16, 16, 16, 3};
    // make some variables
    rc.Add("v1", m_in, scalar_block_size);
    rc.Add("v2", m_out, scalar_block_size);
    rc.Add("v3", m_in, vector_block_size);
    rc.Add("v4", m_out, vector_block_size);
    rc.Add("v5", m_in, scalar_block_size);
    rc.Add("v6", m_out, scalar_block_size);

    WHEN("We initialize all variables to zero") {
      // set them all to zero
      const CellVariableVector<Real> &cv = rc.GetCellVariableVector();
      for (int n = 0; n < cv.size(); n++) {
        ParArrayND<Real> v = cv[n]->data;
        par_for(
            "Initialize variables", DevSpace(), 0, v.GetDim(4) - 1, 0, v.GetDim(3) - 1, 0,
            v.GetDim(2) - 1, 0, v.GetDim(1) - 1,
            KOKKOS_LAMBDA(const int l, const int k, const int j, const int i) {
              v(l, k, j, i) = 0.0;
            });
      }
      THEN("they should sum to zero") {
        using policy4D = Kokkos::MDRangePolicy<Kokkos::Rank<4>>;
        Real total = 0.0;
        for (int n = 0; n < cv.size(); n++) {
          Real sum = 1.0;
          ParArrayND<Real> v = cv[n]->data;
          Kokkos::parallel_reduce(
              policy4D({0, 0, 0, 0},
                       {v.GetDim(4), v.GetDim(3), v.GetDim(2), v.GetDim(1)}),
              KOKKOS_LAMBDA(const int l, const int k, const int j, const int i,
                            Real &vsum) { vsum += v(l, k, j, i); },
              sum);
          total += sum;
        }
        REQUIRE(total == 0.0);
      }
      AND_THEN("we touch the right number of elements") {
        using policy4D = Kokkos::MDRangePolicy<Kokkos::Rank<4>>;
        int total = 0;
        for (int n = 0; n < cv.size(); n++) {
          int sum = 1;
          ParArrayND<Real> v = cv[n]->data;
          Kokkos::parallel_reduce(
              policy4D({0, 0, 0, 0},
                       {v.GetDim(4), v.GetDim(3), v.GetDim(2), v.GetDim(1)}),
              KOKKOS_LAMBDA(const int l, const int k, const int j, const int i,
                            int &cnt) { cnt++; },
              sum);
          total += sum;
        }
        REQUIRE(total == 40960);
      }
    }

    WHEN("we set Independent variables to one") {
      // set "Independent" variables to one
      ContainerIterator<Real> ci(rc, {Metadata::Independent});
      CellVariableVector<Real> &civ = ci.vars;
      for (int n = 0; n < civ.size(); n++) {
        ParArrayND<Real> v = civ[n]->data;
        par_for(
            "Set independent variables", DevSpace(), 0, v.GetDim(4) - 1, 0,
            v.GetDim(3) - 1, 0, v.GetDim(2) - 1, 0, v.GetDim(1) - 1,
            KOKKOS_LAMBDA(const int l, const int k, const int j, const int i) {
              v(l, k, j, i) = 1.0;
            });
      }

      THEN("they should sum appropriately") {
        using policy4D = Kokkos::MDRangePolicy<Kokkos::Rank<4>>;
        Real total = 0.0;
        ContainerIterator<Real> ci(rc, {Metadata::Independent});
        CellVariableVector<Real> &civ = ci.vars;
        for (int n = 0; n < civ.size(); n++) {
          Real sum = 1.0;
          ParArrayND<Real> v = civ[n]->data;
          Kokkos::parallel_reduce(
              policy4D({0, 0, 0, 0},
                       {v.GetDim(4), v.GetDim(3), v.GetDim(2), v.GetDim(1)}),
              KOKKOS_LAMBDA(const int l, const int k, const int j, const int i,
                            Real &vsum) { vsum += v(l, k, j, i); },
              sum);
          total += sum;
        }
        REQUIRE(std::abs(total - 20480.0) < 1.e-14);
      }
    }
  }
}
//Test wrapper to run a function multiple times
template<typename InitFunc,typename PerfFunc>
double performance_test_wrapper(const int n_burn, const int n_perf,
    InitFunc init_func,PerfFunc perf_func){

  //Initialize the timer and test
  Kokkos::Timer timer;
  init_func();

  for( int i_run = 0; i_run < n_burn + n_perf; i_run++){
    if(i_run == n_burn){
      //Burn in time is over, start timing
      Kokkos::fence();
      timer.reset();
    }

    //Run the function timing performance
    perf_func();
  }

  //Time it
  Kokkos::fence();
  double perf_time = timer.seconds();

  //FIXME?
  //Validate results?

  return perf_time;

}


TEST_CASE("Container Iterator Performance",
          "[ContainerIterator][performance]") {


  const int N = 16; //Dimensions of blocks
  const int n_burn = 500; //Num times to burn in before timing
  const int n_perf = 500; //Num times to run while timing

  //Make a raw ParArray4D for closest to bare metal looping
  ParArray4D<Real> raw_array("raw_array",10,N,N,N);

  //Make a function for initializing the raw ParArray4D
  auto init_raw_array = [&]() {
    par_for("Initialize ", DevSpace(),
      0, 10-1,
      0, N-1,
      0, N-1,
      0, N-1,
      KOKKOS_LAMBDA(const int l, const int k, const int j, const int i) {
        raw_array(l,k,j,i) = static_cast<Real>( (l+1)*(k+1)*(j+1)*(i+1) );
      });
  };

  //Test performance iterating over variables (we should aim for this performance)
  double time_raw_array = performance_test_wrapper( n_burn, n_perf,init_raw_array,
    [&](){
      par_for("Raw Array Perf", DevSpace(),
        0, 10-1,
        0, N-1,
        0, N-1,
        0, N-1,
        KOKKOS_LAMBDA(const int l, const int k, const int j, const int i) {
          raw_array(l,k,j,i) *= raw_array(l,k,j,i); //Do something trivial, square each term
        });
    });


  //Make a container for testing performance
  Container<Real> container;
  Metadata m_in({Metadata::Independent});
  Metadata m_out;
  std::vector<int> scalar_block_size {N,N,N};
  std::vector<int> vector_block_size {3,N,N,N};

  // make some variables - 5 in all, 2 3-vectors, total 10 fields
  container.Add("v0",m_in, scalar_block_size);
  container.Add("v1",m_in, scalar_block_size);
  container.Add("v2",m_in, vector_block_size);
  container.Add("v3",m_in, scalar_block_size);
  container.Add("v4",m_in, vector_block_size);

  //Make a function for initializing the container variables
  auto init_container = [&]() {
    const CellVariableVector<Real>& cv = container.GetCellVariableVector();
    for (int n=0; n<cv.size(); n++) {
      ParArrayND<Real> v = cv[n]->data;
      par_for("Initialize variables", DevSpace(),
        0, v.GetDim(4)-1,
        0, v.GetDim(3)-1,
        0, v.GetDim(2)-1,
        0, v.GetDim(1)-1,
        KOKKOS_LAMBDA(const int l, const int k, const int j, const int i) {
          v(l,k,j,i) = static_cast<Real>( (l+1)*(k+1)*(j+1)*(i+1) );
        });
    }
  };

  //Test performance iterating over variables in container
  double time_iterate_variables = performance_test_wrapper( n_burn, n_perf,init_container,
    [&](){
      const CellVariableVector<Real>& cv = container.GetCellVariableVector();
      for (int n=0; n<cv.size(); n++) {
        ParArrayND<Real> v = cv[n]->data;
        par_for("Iterate Variables Perf", DevSpace(),
          0, v.GetDim(4)-1,
          0, v.GetDim(3)-1,
          0, v.GetDim(2)-1,
          0, v.GetDim(1)-1,
          KOKKOS_LAMBDA(const int l, const int k, const int j, const int i) {
            v(l,k,j,i) *= v(l,k,j,i); //Do something trivial, square each term
          });
      }
    });


  //Make a View of Views proof of concept
  const CellVariableVector<Real>& cv = container.GetCellVariableVector();
  Kokkos::View< ParArrayND<Real>* > var_view("var_view",cv.size());
  auto h_var_view = Kokkos::create_mirror_view(var_view);
  for (int n=0; n<cv.size(); n++) {
    h_var_view[n] = cv[n]->data; //Will this behave correctly on the device?
  }
  Kokkos::deep_copy(var_view,h_var_view);
  
  //Use the same function for containers for initializing container variables 

  //Test performance iterating over var_view in one kernel
  double time_view_of_views = performance_test_wrapper( n_burn, n_perf,init_container,
    [&](){
      par_for("View of Views Perf", DevSpace(),
        0, cv.size()-1,
        0, cv[0]->data.GetDim(3)-1,
        0, cv[0]->data.GetDim(2)-1,
        0, cv[0]->data.GetDim(1)-1,
        KOKKOS_LAMBDA(const int n, const int k, const int j, const int i) {
          auto v = var_view(n);
          for(int l = 0; l < v.GetDim(4); l++){
            v(l,k,j,i) *= v(l,k,j,i); //Do something trivial, square each term
          }
        });
    });



  std::cout << "raw_array performance: " << time_raw_array << std::endl;
  std::cout << "iterate_variables performance: " << time_iterate_variables << std::endl;
  std::cout << "iterate_variables/raw_array " << time_iterate_variables/time_raw_array << std::endl;
  std::cout << "view_of_views performance: " << time_view_of_views << std::endl;
  std::cout << "view_of_views/raw_array " << time_view_of_views/time_raw_array << std::endl;

}
