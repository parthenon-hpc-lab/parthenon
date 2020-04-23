
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
using parthenon::Real;


static void setVector( const ParArrayND<Real> &v, const Real &value) {
  par_for(
	  "Initialize variables", DevSpace(), 0, v.GetDim(4) - 1, 0, v.GetDim(3) - 1, 0,
	  v.GetDim(2) - 1, 0, v.GetDim(1) - 1,
	  KOKKOS_LAMBDA(const int l, const int k, const int j, const int i) {
	    v(l, k, j, i) = value;
	  });
}

static Real sumVector( const ParArrayND<Real> &v ) {
  using policy4D = Kokkos::MDRangePolicy<Kokkos::Rank<4>>;
  Real sum = 0.0;
  Kokkos::parallel_reduce(
			  policy4D({0, 0, 0, 0},
				   {v.GetDim(4), v.GetDim(3), v.GetDim(2), v.GetDim(1)}),
			  KOKKOS_LAMBDA(const int l, const int k, const int j, const int i,
					Real &vsum) { vsum += v(l, k, j, i); },
			  sum);
  return sum;
}

static int numElements( const ParArrayND<Real> &v ) {
  using policy4D = Kokkos::MDRangePolicy<Kokkos::Rank<4>>;
  int sum;
  Kokkos::parallel_reduce( policy4D({0, 0, 0, 0},
				    {v.GetDim(4), v.GetDim(3), v.GetDim(2), v.GetDim(1)}),
			   KOKKOS_LAMBDA(const int l, const int k, const int j, const int i,
					 int &cnt) { cnt++; },
			   sum);
  return sum;
}

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
      for (auto &vec : cv ) { setVector(vec->data, 0.0); }

      THEN("they should sum to zero") {
        Real total = 0.0;
        for (auto &vec : cv ) { total += sumVector(vec->data); }
        REQUIRE(total == 0.0);
      }

      AND_THEN("we touch the right number of elements") {
        int nElements = 0;
        for (auto &vec : cv ) { nElements += numElements(vec->data); }
        REQUIRE(nElements == 40960);
      }
    }

    WHEN("we set Independent variables to one") {
      // set "Independent" variables to one
      ContainerIterator<Real> ci(rc, {Metadata::Independent});
      CellVariableVector<Real> &civ = ci.vars;
      for (int n = 0; n < civ.size(); n++) {
        ParArrayND<Real> &v = civ[n]->data;
	setVector(v, 1.0);
      }

      THEN("they should sum appropriately") {
        Real total = 0.0;
        for (int n = 0; n < civ.size(); n++) {
          ParArrayND<Real> &v = civ[n]->data;
	  total += sumVector(v);
        }
        REQUIRE(std::abs(total - 20480.0) < 1.e-14);
      }
    }
  }
}
