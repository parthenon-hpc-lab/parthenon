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
#ifndef ATHENA_HPP_
#define ATHENA_HPP_
//! \file athena.hpp
//  \brief contains Athena++ general purpose types, structures, enums, etc.

// C headers

// C++ headers
#include <cmath>
#include <cstdint>  // std::int64_t

// Kokkos headers
#include <Kokkos_Core.hpp>

// Athena++ headers
#include "athena_arrays.hpp"
#include <defs.hpp>

namespace parthenon {

// primitive type alias that allows code to run with either floats or doubles
#if SINGLE_PRECISION_ENABLED
using Real = float;
#ifdef MPI_PARALLEL
#define MPI_ATHENA_REAL MPI_FLOAT
#endif
#else
using Real = double;
#ifdef MPI_PARALLEL
#define MPI_ATHENA_REAL MPI_DOUBLE
#endif
#endif

#ifdef KOKKOS_ENABLE_CUDA_UVM
typedef Kokkos::CudaUVMSpace     DevSpace;
typedef Kokkos::CudaUVMSpace     HostSpace;
#else
typedef Kokkos::DefaultExecutionSpace     DevSpace;
typedef Kokkos::HostSpace                 HostSpace;
#endif

template<typename T = Real>
using AthenaArray1D = Kokkos::View<T*    , Kokkos::LayoutRight, DevSpace>;
template<typename T = Real>
using AthenaArray2D = Kokkos::View<T**   , Kokkos::LayoutRight, DevSpace>;
template<typename T = Real>
using AthenaArray3D = Kokkos::View<T***  , Kokkos::LayoutRight, DevSpace>;
template<typename T = Real>
using AthenaArray4D = Kokkos::View<T**** , Kokkos::LayoutRight, DevSpace>;
template<typename T = Real>
using AthenaArray5D = Kokkos::View<T*****, Kokkos::LayoutRight, DevSpace>;

typedef Kokkos::TeamPolicy<>               team_policy;
typedef Kokkos::TeamPolicy<>::member_type  member_type;

static struct LoopPatternSimdFor {} loop_pattern_simdfor_tag;
static struct LoopPatternRange {} loop_pattern_range_tag;
static struct LoopPatternMDRange {} loop_pattern_mdrange_tag;
static struct LoopPatternTPX {} loop_pattern_tpx_tag;
static struct LoopPatternTPTTRTVR {} loop_pattern_tpttrtvr_tag;

#ifdef MANUAL1D_LOOP
#define DEFAULT_LOOP_PATTERN loop_pattern_range_tag
#elif defined SIMDFOR_LOOP
#define DEFAULT_LOOP_PATTERN loop_pattern_simdfor_tag
#elif defined MDRANGE_LOOP
#define DEFAULT_LOOP_PATTERN loop_pattern_mdrange_tag
#elif defined TP_INNERX_LOOP
#define DEFAULT_LOOP_PATTERN loop_pattern_tpx_tag
#elif defined TPTTRTVR_LOOP
#define DEFAULT_LOOP_PATTERN loop_pattern_tpttrtvr_tag
#else
#error undefined loop pattern
#endif

#ifdef INNER_TTR_LOOP
#define TPINNERLOOP Kokkos::TeamThreadRange
#elif defined INNER_TVR_LOOP
#define TPINNERLOOP Kokkos::ThreadVectorRange
#else
#define TPINNERLOOP Kokkos::TeamThreadRange
#endif


// 3D default loop pattern
template <typename Function>
inline void par_for(const std::string & NAME,
                       const int & KL, const int & KU,
                       const int & JL, const int & JU,
                       const int & IL, const int & IU,
                       const Function & function) {
  par_for(DEFAULT_LOOP_PATTERN,NAME,KL,KU,JL,JU,IL,IU,function);
}

// 4D default loop pattern
template <typename Function>
inline void par_for(const std::string & NAME,
                       const int & NL, const int & NU,
                       const int & KL, const int & KU,
                       const int & JL, const int & JU,
                       const int & IL, const int & IU,
                       const Function & function) {
  par_for(DEFAULT_LOOP_PATTERN,NAME,NL,NU,KL,KU,JL,JU,IL,IU,function);
}

// 3D loop using Kokkos 1D Range
template <typename Function>
inline void par_for(LoopPatternRange, const std::string & NAME,
                       const int & KL, const int & KU,
                       const int & JL, const int & JU,
                       const int & IL, const int & IU,
                       const Function & function) {
  const int NK = KU - KL + 1;
  const int NJ = JU - JL + 1;
  const int NI = IU - IL + 1;
  const int NKNJNI = NK*NJ*NI;
  const int NJNI = NJ * NI;
  Kokkos::parallel_for(NAME,
    NKNJNI,
    KOKKOS_LAMBDA (const int& IDX) {
    int k = IDX / NJNI;
    int j = (IDX - k*NJNI) / NI;
    int i = IDX - k*NJNI - j*NI;
    k += KL;
    j += JL;
    i += IL;
    function(k,j,i);
    });
}

// 3D loop using MDRange loops
template <typename Function>
inline void par_for(LoopPatternMDRange, const std::string & NAME,
                       const int & KL, const int & KU,
                       const int & JL, const int & JU,
                       const int & IL, const int & IU,
                       const Function & function) {
  Kokkos::parallel_for(NAME,
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
      {KL,JL,IL},{KU+1,JU+1,IU+1}),
    function);
}

// 3D loop using TeamPolicy with single inner loops
template <typename Function>
inline void par_for(LoopPatternTPX, const std::string & NAME,
                       const int & KL, const int & KU,
                       const int & JL, const int & JU,
                       const int & IL, const int & IU,
                       const Function & function) {
  const int NK = KU - KL + 1;
  const int NJ = JU - JL + 1;
  const int NKNJ = NK * NJ;
  Kokkos::parallel_for(NAME,
    team_policy (NKNJ, Kokkos::AUTO,PAR_VECTOR_LENGTH),
    KOKKOS_LAMBDA (member_type team_member) {
      const int k = team_member.league_rank() / NJ + KL;
      const int j = team_member.league_rank() % NJ + JL;
      Kokkos::parallel_for(
        TPINNERLOOP<>(team_member,IL,IU+1),
        [&] (const int i) {
          function(k,j,i);
        });
    });
}

// 3D loop using TeamPolicy with nested TeamThreadRange and ThreadVectorRange
template <typename Function>
inline void par_for(LoopPatternTPTTRTVR, const std::string & NAME,
                       const int & KL, const int & KU,
                       const int & JL, const int & JU,
                       const int & IL, const int & IU,
                       const Function & function) {

  const int NK = KU - KL + 1;
  Kokkos::parallel_for(NAME,
    team_policy (NK, Kokkos::AUTO,PAR_VECTOR_LENGTH),
    KOKKOS_LAMBDA (member_type team_member) {
      const int k = team_member.league_rank() + KL;
      Kokkos::parallel_for(
        Kokkos::TeamThreadRange<>(team_member,JL,JU+1),
        [&] (const int j) {
          Kokkos::parallel_for(
            Kokkos::ThreadVectorRange<>(team_member,IL,IU+1),
            [&] (const int i) {
              function(k,j,i);
            });
        });
    });
}

// 3D loop using SIMD FOR loops
template <typename Function>
inline void par_for(LoopPatternSimdFor, const std::string & NAME,
                       const int & KL, const int & KU,
                       const int & JL, const int & JU,
                       const int & IL, const int & IU,
                       const Function & function) {
  Kokkos::Profiling::pushRegion(NAME);
  for (auto k = KL; k <= KU; k++)
    for (auto j = JL; j <= JU; j++)
      #pragma omp simd
      for (auto i = IL; i <= IU; i++)
        function(k,j,i);
  Kokkos::Profiling::popRegion();
}

// 4D loop using Kokkos 1D Range
template <typename Function>
inline void par_for(LoopPatternRange, const std::string & NAME,
                       const int NL, const int NU,
                       const int KL, const int KU,
                       const int JL, const int JU,
                       const int IL, const int IU,
                       const Function & function) {
  const int NN = (NU) - (NL) + 1;
  const int NK = (KU) - (KL) + 1;
  const int NJ = (JU) - (JL) + 1;
  const int NI = (IU) - (IL) + 1;
  const int NNNKNJNI = NN*NK*NJ*NI;
  const int NKNJNI = NK*NJ*NI;
  const int NJNI = NJ * NI;
  Kokkos::parallel_for(NAME,
    NNNKNJNI,
    KOKKOS_LAMBDA (const int& IDX) {
    int n = IDX / NKNJNI;
    int k = (IDX - n*NKNJNI) / NJNI;
    int j = (IDX - n*NKNJNI - k*NJNI) / NI;
    int i = IDX - n*NKNJNI - k*NJNI - j*NI;
    n += (NL);
    k += (KL);
    j += (JL);
    i += (IL);
    function(n,k,j,i);
    });
}

// 4D loop using MDRange loops
template <typename Function>
inline void par_for(LoopPatternMDRange, const std::string & NAME,
                       const int NL, const int NU,
                       const int KL, const int KU,
                       const int JL, const int JU,
                       const int IL, const int IU,
                       const Function & function) {
  Kokkos::parallel_for(NAME,
    Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
      {NL,KL,JL,IL},{NU+1,KU+1,JU+1,IU+1}),
    function);
}

// 4D loop using TeamPolicy loops
template <typename Function>
inline void par_for(LoopPatternTPX, const std::string & NAME,
                       const int NL, const int NU,
                       const int KL, const int KU,
                       const int JL, const int JU,
                       const int IL, const int IU,
                       const Function & function) {
  const int NN = NU - NL + 1;
  const int NK = KU - KL + 1;
  const int NJ = JU - JL + 1;
  const int NKNJ = NK * NJ;
  const int NNNKNJ = NN * NK * NJ;
  Kokkos::parallel_for(NAME,
    team_policy (NNNKNJ, Kokkos::AUTO,PAR_VECTOR_LENGTH),
    KOKKOS_LAMBDA (member_type team_member) {
      int n = team_member.league_rank() / NKNJ;
      int k = (team_member.league_rank() - n*NKNJ) / NJ;
      int j = team_member.league_rank() - n*NKNJ - k*NJ + JL;
      n += NL;
      k += KL;
      Kokkos::parallel_for(
        TPINNERLOOP<>(team_member,IL,IU+1),
        [&] (const int i) {
          function(n,k,j,i);
        });
    });
}

// 4D loop using TeamPolicy with nested TeamThreadRange and ThreadVectorRange
template <typename Function>
inline void par_for(LoopPatternTPTTRTVR, const std::string & NAME,
                       const int NL, const int NU,
                       const int KL, const int KU,
                       const int JL, const int JU,
                       const int IL, const int IU,
                       const Function & function) {
  const int NN = NU - NL + 1;
  const int NK = KU - KL + 1;
  const int NNNK = NN * NK;
  Kokkos::parallel_for(NAME,
    team_policy (NNNK, Kokkos::AUTO,PAR_VECTOR_LENGTH),
    KOKKOS_LAMBDA (member_type team_member) {
      int n = team_member.league_rank() / NK + NL;
      int k = team_member.league_rank() % NK + KL;
      Kokkos::parallel_for(
        Kokkos::TeamThreadRange<>(team_member,JL,JU+1),
        [&] (const int j) {
          Kokkos::parallel_for(
            Kokkos::ThreadVectorRange<>(team_member,IL,IU+1),
            [&] (const int i) {
              function(n,k,j,i);
            });
        });
    });
}

// 4D loop using SIMD FOR loops
template <typename Function>
inline void par_for(LoopPatternSimdFor, const std::string & NAME,
                       const int NL, const int NU,
                       const int KL, const int KU,
                       const int JL, const int JU,
                       const int IL, const int IU,
                       const Function & function) {
  Kokkos::Profiling::pushRegion(NAME);
  for (auto n = NL; n <= NU; n++)
    for (auto k = KL; k <= KU; k++)
      for (auto j = JL; j <= JU; j++)
        #pragma omp simd
        for (auto i = IL; i <= IU; i++)
          function(n,k,j,i);
  Kokkos::Profiling::popRegion();
}

// for OpenMP 4.0 SIMD vectorization, control width of SIMD lanes
#if defined(__AVX512F__)
#define SIMD_WIDTH 8
#elif defined(__AVX__)
#define SIMD_WIDTH 4
#elif defined(__SSE2__)
#define SIMD_WIDTH 2
#else
#define SIMD_WIDTH 4
#endif

#define CACHELINE_BYTES 64

// forward declarations needed for function pointer type aliases
class MeshBlock;
class Coordinates;
class ParameterInput;
class HydroDiffusion;
class FieldDiffusion;

//--------------------------------------------------------------------------------------
//! \struct LogicalLocation
//  \brief stores logical location and level of MeshBlock

struct LogicalLocation { // aggregate and POD type
  // These values can exceed the range of std::int32_t even if the root grid has only a
  // single MeshBlock if >30 levels of AMR are used, since the corresponding max index =
  // 1*2^31 > INT_MAX = 2^31 -1 for most 32-bit signed integer type impelementations
  std::int64_t lx1, lx2, lx3;
  int level;

  // operators useful for sorting
  bool operator==(LogicalLocation &ll) {
    return ((ll.level == level) && (ll.lx1 == lx1) && (ll.lx2 == lx2) && (ll.lx3 == lx3));
  }
  static bool Lesser(const LogicalLocation &left, const LogicalLocation &right) {
    return left.level < right.level;
  }
  static bool Greater(const LogicalLocation & left, const LogicalLocation &right) {
    return left.level > right.level;
  }
};

//----------------------------------------------------------------------------------------
//! \struct RegionSize
//  \brief physical size and number of cells in a Mesh or a MeshBlock

struct RegionSize {  // aggregate and POD type; do NOT reorder member declarations:
  Real x1min, x2min, x3min;
  Real x1max, x2max, x3max;
  Real x1rat, x2rat, x3rat; // ratio of dxf(i)/dxf(i-1)
  // the size of the root grid or a MeshBlock should not exceed std::int32_t limits
  int nx1, nx2, nx3;        // number of active cells (not including ghost zones)
};

//---------------------------------------------------------------------------------------
//! \struct FaceField
//  \brief container for face-centered fields

struct FaceField {
  AthenaArray<Real> x1f, x2f, x3f;
  FaceField() = default;
  FaceField(int ncells3, int ncells2, int ncells1,
            AthenaArray<Real>::DataStatus init=AthenaArray<Real>::DataStatus::allocated) :
      x1f(ncells3, ncells2, ncells1+1, init), x2f(ncells3, ncells2+1, ncells1, init),
      x3f(ncells3+1, ncells2, ncells1, init) {}
};

//----------------------------------------------------------------------------------------
//! \struct EdgeField
//  \brief container for edge-centered fields

struct EdgeField {
  AthenaArray<Real> x1e, x2e, x3e;
  EdgeField() = default;
  EdgeField(int ncells3, int ncells2, int ncells1,
            AthenaArray<Real>::DataStatus init=AthenaArray<Real>::DataStatus::allocated) :
      x1e(ncells3+1, ncells2+1, ncells1, init), x2e(ncells3+1, ncells2, ncells1+1, init),
      x3e(ncells3, ncells2+1, ncells1+1, init) {}
};

//----------------------------------------------------------------------------------------
// enums used everywhere
// (not specifying underlying integral type (C++11) for portability & performance)

// TODO(felker): C++ Core Guidelines Enum.5: Don’t use ALL_CAPS for enumerators
// (avoid clashes with preprocessor macros). Enumerated type definitions in this file and:
// io_wrapper.hpp, bvals.hpp, hydro_diffusion.hpp, field_diffusion.hpp,
// task_list.hpp, ???

//------------------
// named, weakly typed / unscoped enums:
//------------------

// needed for arrays dimensioned over grid directions
// enumerator type only used in Mesh::EnrollUserMeshGenerator()
enum CoordinateDirection {X1DIR=0, X2DIR=1, X3DIR=2};

//------------------
// strongly typed / scoped enums (C++11):
//------------------
// KGF: Except for the 2x MG* enums, these may be unnessary w/ the new class inheritance
// Now, only passed to BoundaryVariable::InitBoundaryData(); could replace w/ bool switch
enum class BoundaryQuantity {cc, fc, cc_flcor, fc_flcor};
enum class BoundaryCommSubset {mesh_init, gr_amr, all};
// TODO(felker): consider generalizing/renaming to QuantityFormulation
enum class UserHistoryOperation {sum, max, min};

//----------------------------------------------------------------------------------------
// function pointer prototypes for user-defined modules set at runtime

using BValFunc = void (*)(
    MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
    Real time, Real dt,
    int is, int ie, int js, int je, int ks, int ke, int ngh);
using AMRFlagFunc = int (*)(MeshBlock *pmb);
using MeshGenFunc = Real (*)(Real x, RegionSize rs);
using SrcTermFunc = void (*)(
    MeshBlock *pmb, const Real time, const Real dt,
    const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc, AthenaArray<Real> &cons);
using TimeStepFunc = Real (*)(MeshBlock *pmb);
using HistoryOutputFunc = Real (*)(MeshBlock *pmb, int iout);
using MetricFunc = void (*)(
    Real x1, Real x2, Real x3, ParameterInput *pin,
    AthenaArray<Real> &g, AthenaArray<Real> &g_inv,
    AthenaArray<Real> &dg_dx1, AthenaArray<Real> &dg_dx2, AthenaArray<Real> &dg_dx3);
using MGBoundaryFunc = void (*)(
    AthenaArray<Real> &dst,Real time, int nvar,
    int is, int ie, int js, int je, int ks, int ke, int ngh,
    Real x0, Real y0, Real z0, Real dx, Real dy, Real dz);
using ViscosityCoeffFunc = void (*)(
    HydroDiffusion *phdif, MeshBlock *pmb,
    const  AthenaArray<Real> &w, const AthenaArray<Real> &bc,
    int is, int ie, int js, int je, int ks, int ke);
using ConductionCoeffFunc = void (*)(
    HydroDiffusion *phdif, MeshBlock *pmb,
    const AthenaArray<Real> &w, const AthenaArray<Real> &bc,
    int is, int ie, int js, int je, int ks, int ke);
using FieldDiffusionCoeffFunc = void (*)(
    FieldDiffusion *pfdif, MeshBlock *pmb,
    const AthenaArray<Real> &w,
    const AthenaArray<Real> &bmag,
    int is, int ie, int js, int je, int ks, int ke);

}
#endif // ATHENA_HPP_
