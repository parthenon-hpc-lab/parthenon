
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
using parthenon::MetadataFlag;

/*template <typename T>
Kokkos::View<T*> make_view_of_type(T orig_view, std::string label, int size) {
  Kokkos::View<T*> new_view(label, size);
  return std::move(new_view);
}*/

template <typename T>
class OrigFlattenedIterator {
 public:
  OrigFlattenedIterator(T orig_view, std::string label, int size) : v(label, size) {}
  template <class... Args>
  KOKKOS_FORCEINLINE_FUNCTION
  auto& operator() (const int n, Args... args) const {
    return v(n)(std::forward<Args>(args)...);
  }
  auto create_mirror_view() {
    return Kokkos::create_mirror_view(v);
  }
  template <typename U>
  void deep_copy(U h_var_view) {
    Kokkos::deep_copy(v, h_var_view);
  }
  KOKKOS_FORCEINLINE_FUNCTION
  auto GetDim(const int i) {
    if (i==4) return v.extent_int(0);
    else return v(0).extent_int(3-i);
  }

 private:
  Kokkos::View<T*> v;
};


// TODO(JMM): Holding a device view of views
// and a host mirror just to enable `GetDim`
// is not an ideal solution. A better solution
// would be to store this shape in a private variable.
template <typename T, typename TH>
class FlattenedIterator {
 public:
  FlattenedIterator(T view, TH view_host) : v(view), v_h(view_host) {}
  KOKKOS_FORCEINLINE_FUNCTION
  auto& operator() (const int n) {
    return v(n);
  }
  template <class... Args>
  KOKKOS_FORCEINLINE_FUNCTION
  auto& operator() (const int n, Args... args) const {
    return v(n)(std::forward<Args>(args)...);
  }
  KOKKOS_FORCEINLINE_FUNCTION
  auto GetDimDevice(const int i) {
    if (i==4) return v.extent_int(0);
    else return v(0).extent_int(3-i);
  }
  auto GetDimHost(const int i) {
    if (i==4) return v_h.extent_int(0);
    else return v_h(0).extent_int(3-i);
  }
 private:
  T v;
  TH v_h;
};


template <typename T>
auto MakeIterator(const Container<T> &c,
                    const std::vector<MetadataFlag> &flagVector) {
  CellVariableVector<T> vars;
  for (const auto& v : c.GetCellVariableVector()) {
    if (v->metadata().AnyFlagsSet(flagVector)) {
      vars.push_back(v);
    }
  }
  for (const auto& v : c.GetSparseVector()) {
    if (v->metadata().AnyFlagsSet(flagVector)) {
      auto& svec = v->GetVector();
      vars.insert(vars.end(), svec.begin(), svec.end());
    }
  }

  // count up the size
  int vsize = 0;
  for (const auto& v : vars) {
    vsize += v->GetDim(6)*v->GetDim(5)*v->GetDim(4);
  }

  auto array = vars[0]->data;
  auto slice_type = array.Get(0,0,0);
  auto cv = Kokkos::View<decltype(slice_type)*>("ContainerIterator::cv_",vsize);
  auto host_view = Kokkos::create_mirror_view(cv);
  int vindex = 0;
  for (const auto& v : vars) {
    for (int k=0; k<v->GetDim(6); k++) {
      for (int j=0; j<v->GetDim(5); j++) {
        for (int i=0; i<v->GetDim(4); i++) {
          host_view(vindex++) = v->data.Get(k,j,i);
        }
      }
    }
  }
  Kokkos::deep_copy(cv, host_view);
  return FlattenedIterator<decltype(cv),decltype(host_view)>(cv,host_view);
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


  const int N = 32; //Dimensions of blocks
  const int Nvar = 10;
  const int n_burn = 500; //Num times to burn in before timing
  const int n_perf = 500; //Num times to run while timing

  //Make a raw ParArray4D for closest to bare metal looping
  ParArrayND<Real> raw_array("raw_array",Nvar,N,N,N);

  //Make a function for initializing the raw ParArray4D
  auto init_raw_array = [&]() {
    par_for("Initialize ", DevSpace(),
      0, Nvar-1,
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
        0, Nvar-1,
        0, N-1,
        0, N-1,
        0, N-1,
        KOKKOS_LAMBDA(const int l, const int k, const int j, const int i) {
          raw_array(l,k,j,i) *= raw_array(l,k,j,i); //Do something trivial, square each term
        });
    });
    //std::cout << "val = " << raw_array(3,2,3,4) << std::endl;

  //Make a container for testing performance
  Container<Real> container;
  Metadata m_in({Metadata::Independent});
  Metadata m_out;
  std::vector<int> scalar_block_size {N,N,N};
  std::vector<int> vector_block_size {N,N,N,3};

  // make some variables - 5 in all, 2 3-vectors, total 10 fields
  container.Add("v0",m_in, scalar_block_size);
  container.Add("v1",m_in, scalar_block_size);
  container.Add("v2",m_in, vector_block_size);
  container.Add("v3",m_in, scalar_block_size);
  container.Add("v4",m_in, vector_block_size);
  container.Add("v5",m_in, scalar_block_size);

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
  double time_iterate_variables = 0.0;

  //Test performance iterating over variables in container
  time_iterate_variables = performance_test_wrapper( n_burn, n_perf,init_container,
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

/*
    const CellVariableVector<Real> &cv = container.GetCellVariableVector();
    std::cout << "val = " << cv[2]->data(1,2,3,4) << std::endl;

  // count the size
  int vsize = 0;
  for (int n=0; n<cv.size(); n++) vsize += cv[n]->GetDim(4);
  //auto var_view = make_view_of_type<>(cv[0]->data.Get(0,0,1), "var_view", vsize);
  auto var_view = OrigFlattenedIterator<decltype(cv[0]->data.Get(0,0,0))>(cv[0]->data.Get(0,0,0), "var_view", vsize);
  //auto h_var_view = Kokkos::create_mirror_view(var_view);
  auto h_var_view = var_view.create_mirror_view();
  int vindex = 0;
  for (int n=0; n<cv.size(); n++) {
    for (int l=0; l<cv[n]->GetDim(4); l++) {
      h_var_view(vindex++) = cv[n]->data.Get(0,0,l);
    }
  }
  //var_view.DeepCopy(h_var_view);
  var_view.deep_copy<>(h_var_view);
  //Kokkos::deep_copy(var_view, h_var_view);
*/

  auto var_view = MakeIterator<Real>(container, {Metadata::Independent});

  //std::cout << "DIMS: " << var_view.GetDim(4) << " " << var_view.GetDim(3)
  //          << " " << var_view.GetDim(2) << " " << var_view.GetDim(1)
  //          << std::endl;

  auto init_view_of_views = [&]() {
    par_for("Initialize ", DevSpace(),
      0, var_view.GetDimHost(4)-1,
      0, var_view.GetDimHost(3)-1,
      0, var_view.GetDimHost(2)-1,
      0, var_view.GetDimHost(1)-1,
      KOKKOS_LAMBDA(const int l, const int k, const int j, const int i) {
        //auto& v = var_view(l);
        var_view(l,k,j,i) = static_cast<Real>( (l+1)*(k+1)*(j+1)*(i+1) );
      });
  };

  //Test performance iterating over variables (we should aim for this performance)
  double time_view_of_views = performance_test_wrapper( n_burn, n_perf,init_view_of_views,
    [&](){
      par_for("Flat Container Array Perf", DevSpace(),
      0, var_view.GetDimHost(4)-1,
      0, var_view.GetDimHost(3)-1,
      0, var_view.GetDimHost(2)-1,
      0, var_view.GetDimHost(1)-1,
        KOKKOS_LAMBDA(const int l,const int k, const int j, const int i) {
          //auto& v = var_view(l);
          var_view(l,k,j,i) *= var_view(l,k,j,i); //Do something trivial, square each term
        });
    });

    //std::cout << "val = " << var_view(3,2,3,4) << std::endl;
/*


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
*/
  std::cout << "raw_array performance: " << time_raw_array << std::endl;
  std::cout << "iterate_variables performance: " << time_iterate_variables << std::endl;
  std::cout << "iterate_variables/raw_array " << time_iterate_variables/time_raw_array << std::endl;
  std::cout << "view_of_views performance: " << time_view_of_views << std::endl;
  std::cout << "view_of_views/raw_array " << time_view_of_views/time_raw_array << std::endl;

}
