//========================================================================================
// (C) (or copyright) 2025. Triad National Security, LLC. All rights reserved.
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

#ifndef TENSORS_TENSORS_HPP_
#define TENSORS_TENSORS_HPP_

#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "basic_types.hpp"
#include "defs.hpp"
#include "kokkos_abstraction.hpp"
#include "parthenon_arrays.hpp"
#include "utils/object_pool.hpp"

namespace parthenon {

namespace tensors {

KOKKOS_INLINE_FUNCTION
Real safe_sqrt(const Real a) { return std::sqrt(std::max(a, Real(0.))); }

// unmanaged 2d and 3d kokkos view spiel
using DevSpace = parthenon::DevMemSpace;
using ScratchSpace = Kokkos::ScratchMemorySpace<DevSpace>;
using View2DUnmanaged = Kokkos::View<double **, Kokkos::LayoutRight, DevSpace,
                                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
// using View2DUnmanaged = Kokkos::View<double **, Kokkos::LayoutRight, ScratchSpace,
//                                      Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
using View3DUnmanaged = Kokkos::View<double ***, Kokkos::LayoutRight, DevSpace,
                                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

constexpr const std::size_t NINDICES = 3;
using shape_t = Kokkos::View<std::size_t[NINDICES]>;
using shape_host_t = typename Kokkos::View<std::size_t[NINDICES]>::host_mirror_type;

// JMM: We need a TensorCoreHost type and a TensorCoreDevice type
// because the reference counting for the cores is done on host, and
// so a host array needs to stick around. But if you copy that array
// to device, bad things happen with Kokkos. The performant solution
// is to hang all the views in TensorCoreHost and then make
// TensorCoreDevice essentially an unmanaged view that is a "device
// context" for the host object, similar to how swarms
// work. Book-keeping happens on host. So TensorCoreHost is reference
// counted and works with the memory pool. TensorCoreDevice has device
// access.
using pool_t = ObjectPool<ParArray1DRaw<Real>>;
using pool_map_t = ObjectPoolMap<ParArray1DRaw<Real>>;
using core_data_host_t = Kokkos::View<pool_t::owner_t **, LayoutWrapper, HostMemSpace>;
using core_data_device_t = Kokkos::View<pool_t::weak_t **, LayoutWrapper, DevMemSpace>;
// I'm actually not sure why this needs to be unmanaged memory on
// device. Kokkos view of views works in other contexts, but somehow
// here, making this kokkos managed memory causes Kokkos to lock
// up. Something to do ensuring TensorCoreDevice is trivially
// copyable.
using core_data_device_unmanaged_t =
    Kokkos::View<pool_t::weak_t **, LayoutWrapper, DevMemSpace, MemUnmanaged>;
/* TODO(JMM): If we wanted to generalize this machinery to an
   arbitrary number of tensor indices, we would need an array to
   track index ordering, but we could use variadic templates to
   express constructors and operator().
*/

class TensorTrain; // Forward declaration

class GramSVDStorage {

  using ExecSpace = parthenon::DevExecSpace;
  using ScratchSpace = Kokkos::ScratchMemorySpace<ExecSpace>;

  // TODO(@SWJ): also make the RealCore a scratchpad1d because why not, the views wrap it as 3d
  using RealCoreStorage = ScratchPad3D<Real>;
  using RealCoreView = Kokkos::View<Real ***, Kokkos::LayoutStride, ScratchSpace,
                                    Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

  // TODO(@SWJ): make the RealMat a scratchpad1d because why not, the views wrap it as 2d
  using RealMat = ScratchPad2D<Real>;
  using RealVec = ScratchPad1D<Real>;
  using IntVec = ScratchPad1D<int>;
  using SizeTVec = ScratchPad1D<std::size_t>;

  using GramStorage = ScratchPad1D<Real>;
  using GramMatView = Kokkos::View<Real **, Kokkos::LayoutRight, ScratchSpace,
                                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

 public:
  // ============================================================
  // Enumerations
  // ============================================================

  enum RealCoreName { CTmp_core, NumRealCores };

  enum RealMatrixName {
    A_mat,
    M_mat,
    SVDU_mat,
    SVDV_mat,
    EVL_mat,
    EVR_mat,
    NumRealMatrices
  };

  enum RealVectorName { EvalL_vec, EvalR_vec, SVDS_vec, NumRealVecs };
  enum IntVectorName { KeepFlags_vec, ModeMap_vec, NumIntVecs };
  enum RealAlgoVecName { EVDRealScratch_vec, NumRealAlgoVecs };
  enum SizeTAlgoVecName { EVDSizeTScratch_vec, NumSizeTAlgoVecs };

  // ============================================================
  // Resize rank-sized views
  // ============================================================

  // TODO(@SWJ) these should probably really be Kokkos::Views with LayoutRight
  // and specified dimensions so that we are *really* accessing contiguous
  // memory. Subviews preserve the stride of the original maximally sized 2d
  // array
  KOKKOS_INLINE_FUNCTION
  void ResizeRankViews(int Rn, int evd_scratch_n) {
    for (int i = 0; i < NumRealMatrices; ++i)
      real_mats_view_[i] = Kokkos::subview(real_mats_storage_[i], std::make_pair(-1, Rn),
                                           std::make_pair(0, Rn));
      // real_mats_view_[i] = View2DUnmanaged(&real_mats_storage_[i](0,0), Rn, Rn);

    for (int i = 0; i < NumRealVecs; ++i)
      real_vecs_view_[i] = Kokkos::subview(real_vecs_storage_[i], std::make_pair(0, Rn));

    for (int i = 0; i < NumIntVecs; ++i)
      int_vecs_view_[i] = Kokkos::subview(int_vecs_storage_[i], std::make_pair(0, Rn));

    for (int i = 0; i < NumRealAlgoVecs; ++i)
      real_algo_vecs_view_[i] =
          Kokkos::subview(real_algo_vecs_storage_[i], std::make_pair(0, evd_scratch_n));

    for (int i = 0; i < NumSizeTAlgoVecs; ++i)
      sizet_algo_vecs_view_[i] =
          Kokkos::subview(sizet_algo_vecs_storage_[i], std::make_pair(0, evd_scratch_n));
  }

  // ============================================================
  // Resize temp core
  // ============================================================

  KOKKOS_INLINE_FUNCTION
  void ResizeCoreView(int Rl, int PIn, int Rr) {
    real_cores_view_[CTmp_core] =
        Kokkos::subview(real_cores_storage_[CTmp_core], std::make_pair(0, Rl),
                        std::make_pair(0, PIn), std::make_pair(0, Rr));
  }

  // ============================================================
  // Gram accessors (packed storage)
  // ============================================================

  KOKKOS_INLINE_FUNCTION
  GramMatView GL(int n) {
    int offset = gram_offsets_(n);
    int Rn = gram_sizes_(n);
    Real *ptr = GL_storage_.data() + offset;
    return GramMatView(ptr, Rn, Rn);
  }

  KOKKOS_INLINE_FUNCTION
  GramMatView GR(int n) {
    int offset = gram_offsets_(n);
    int Rn = gram_sizes_(n);
    Real *ptr = GR_storage_.data() + offset;
    return GramMatView(ptr, Rn, Rn);
  }

  // ============================================================
  // Other accessors
  // ============================================================

  KOKKOS_INLINE_FUNCTION RealCoreView &CTmp() { return real_cores_view_[CTmp_core]; }

  KOKKOS_INLINE_FUNCTION RealMat &A() { return real_mats_view_[A_mat]; }

  KOKKOS_INLINE_FUNCTION RealMat &M() { return real_mats_view_[M_mat]; }

  KOKKOS_INLINE_FUNCTION RealMat &SVDU() { return real_mats_view_[SVDU_mat]; }

  KOKKOS_INLINE_FUNCTION RealMat &SVDV() { return real_mats_view_[SVDV_mat]; }

  KOKKOS_INLINE_FUNCTION RealMat &EVL() { return real_mats_view_[EVL_mat]; }

  KOKKOS_INLINE_FUNCTION RealMat &EVR() { return real_mats_view_[EVR_mat]; }

  KOKKOS_INLINE_FUNCTION RealVec &EvalL() { return real_vecs_view_[EvalL_vec]; }

  KOKKOS_INLINE_FUNCTION RealVec &EvalR() { return real_vecs_view_[EvalR_vec]; }

  KOKKOS_INLINE_FUNCTION RealVec &SVDS() { return real_vecs_view_[SVDS_vec]; }

  KOKKOS_INLINE_FUNCTION IntVec &KeepFlags() { return int_vecs_view_[KeepFlags_vec]; }

  KOKKOS_INLINE_FUNCTION IntVec &ModeMap() { return int_vecs_view_[ModeMap_vec]; }

  KOKKOS_INLINE_FUNCTION RealVec &EVDRealScratch() {
    return real_algo_vecs_view_[EVDRealScratch_vec];
  }

  KOKKOS_INLINE_FUNCTION SizeTVec &EVDSizeTScratch() {
    return sizet_algo_vecs_view_[EVDSizeTScratch_vec];
  }

  // Constructor - defined in cpp
  static size_t GetScratchSize(const TensorTrain &TT, int evd_scratch_max);

  // Get storage requirements - defined in cpp
  KOKKOS_INLINE_FUNCTION
  GramSVDStorage(ScratchSpace ts, const TensorTrain &TT, int evd_scratch_max_);

  KOKKOS_INLINE_FUNCTION
  int CleanAndCountNonZeroEigenValues(RealMat &EVs, RealVec &EVals, const int Rn,
                                      const Real eps) {
    int nnz_eig = 0;

    Real Lambdamax{0.};
    for (int i = 0; i < Rn; i++) {
      Lambdamax = std::max(Lambdamax, EVals(i));
    }

    for (int i = 0; i < Rn; i++) {
      if (EVals(i) < eps * Lambdamax) {
        EVals(i) = 0.;
        for (int j = 0; j < Rn; j++) {
          EVs(j, i) = 0.;
        }
      } else {
        nnz_eig += 1;
      }
    }
    return nnz_eig;
  }

  KOKKOS_INLINE_FUNCTION
  void PrintRealVec(RealVec &V, const int Rn) {
    for (int i = 0; i < Rn; i++)
      printf("%e ", V(i));
    printf("\n");
    printf("\n");
  }

  KOKKOS_INLINE_FUNCTION
  void PrintRealMat(RealMat &M, const int Rn) {
    for (int i = 0; i < Rn; ++i) {
      for (int j = 0; j < Rn; ++j) {
        printf("  %12.5e", M(i, j));
      }
      printf("\n");
      printf("\n");
    }
  }

  KOKKOS_INLINE_FUNCTION
  void ComputeSVD(const int Rn, const int nnzL, const int nnzR);

 private:
  int RMax;
  int PIMax;
  int Ngram_;
  int evd_scratch_max;

  std::array<RealCoreStorage, NumRealCores> real_cores_storage_;
  std::array<RealCoreView, NumRealCores> real_cores_view_;

  std::array<RealMat, NumRealMatrices> real_mats_storage_;
  std::array<RealMat, NumRealMatrices> real_mats_view_;

  std::array<RealVec, NumRealVecs> real_vecs_storage_;
  std::array<RealVec, NumRealVecs> real_vecs_view_;

  std::array<IntVec, NumIntVecs> int_vecs_storage_;
  std::array<IntVec, NumIntVecs> int_vecs_view_;

  std::array<RealVec, NumRealAlgoVecs> real_algo_vecs_storage_;
  std::array<RealVec, NumRealAlgoVecs> real_algo_vecs_view_;

  std::array<SizeTVec, NumSizeTAlgoVecs> sizet_algo_vecs_storage_;
  std::array<SizeTVec, NumSizeTAlgoVecs> sizet_algo_vecs_view_;

  GramStorage GL_storage_;
  GramStorage GR_storage_;

  ScratchPad1D<int> gram_offsets_;
  ScratchPad1D<int> gram_sizes_;
};

class TensorCoreHost;
class TensorCoreDevice {
  friend class TensorCoreHost;

 public:
  TensorCoreDevice() = default;

  KOKKOS_INLINE_FUNCTION
  Real &operator()(int iL, int ic, int iR) const { return data_device_(iL, iR)[ic]; }

  KOKKOS_INLINE_FUNCTION
  auto GetShape() const { return shape_; }
  KOKKOS_INLINE_FUNCTION
  auto GetRanks() const { return std::make_pair(shape_[0], shape_[2]); }
  KOKKOS_INLINE_FUNCTION
  std::size_t GetLeftRank() const { return shape_[0]; }
  KOKKOS_INLINE_FUNCTION
  std::size_t GetRightRank() const { return shape_[2]; }
  KOKKOS_INLINE_FUNCTION
  std::size_t GetPhysicalIndexSize() const { return shape_[1]; }

  // set shape on device
  KOKKOS_FUNCTION
  void SetShape(const int rL, const int PIS, const int rR) const {
    shape_[0] = rL;
    shape_[1] = PIS;
    shape_[2] = rR;
  }

  // Actual constructor is private so that it can only be called from
  // TensorCoreHost
 private:
  explicit TensorCoreDevice(const core_data_device_unmanaged_t &device_data,
                            shape_t shape)
      : data_device_(device_data), shape_(shape) {}
  shape_t shape_;
  core_data_device_unmanaged_t data_device_;
}; // Class TensorCoreDevice

class TensorCoreHost {
 public:
  TensorCoreHost() = default;
  TensorCoreHost(pool_map_t &pool, const std::size_t rL, const std::size_t c,
                 const std::size_t rR) {
    shape_ = shape_t("shape of tensor core (device)");
    shape_host_ = shape_host_t("shape of tensor core (host)");
    shape_host_[0] = rL;
    shape_host_[1] = c;
    shape_host_[2] = rR;
    Kokkos::deep_copy(shape_, shape_host_);

    // Kokkos view of views, the destructor for the view of views must
    // happen on host, not device. This enforces that.
    data_host_ =
        core_data_host_t(ViewOfViewAlloc<HostMemSpace>("tensor core host"), rL, rR);
    data_device_ = core_data_device_t(ViewOfViewAlloc("tensor core device"), rL, rR);

    // construct data object 1d arrays on host, assigning memory from the pool
    for (std::size_t iL = 0; iL < rL; iL++) {
      for (std::size_t iR = 0; iR < rR; iR++) {
        data_host_(iL, iR) = pool.GetPool(c).Get();
      }
    }
    // This deep copy doesn't work because types are mismatched between host and device
    // Kokkos::pdeep_copy(data_device_, data_host_);
    
    // Instead:
    // Create a host mirror of the device view
    auto data_device_mirror = Kokkos::create_mirror_view(data_device_);

    // Fill the mirror from the host view
    for (std::size_t iL = 0; iL < rL; iL++) {
      for (std::size_t iR = 0; iR < rR; iR++) {
        data_device_mirror(iL, iR) = data_host_(iL, iR);
      }
    }

    // Copy mirror to device
    Kokkos::deep_copy(data_device_, data_device_mirror);
  }

  Real &operator()(int iL, int ic, int iR) const { return data_host_(iL, iR)[ic]; }

  TensorCoreDevice GetOnDevice() const {
    return TensorCoreDevice(
        core_data_device_unmanaged_t(data_device_.data(), GetLeftRank(), GetRightRank()),
        shape_);
  }

  auto GetShape() const { return shape_host_; }
  auto GetRanks() const { return std::make_pair(shape_host_[0], shape_host_[2]); }
  std::size_t GetLeftRank() const { return shape_host_[0]; }
  std::size_t GetRightRank() const { return shape_host_[2]; }
  std::size_t GetPhysicalIndexSize() const { return shape_host_[1]; }

  // if the shape_ array was modified on device, shrink to that
  // size.
  void ResizeToNewShape() {
    Kokkos::deep_copy(shape_host_, shape_);
    // TODO(JMM) Currently assumes the new shape is less than the old
    // one. Maybe we want to relax this
    PARTHENON_REQUIRE_THROWS(shape_host_(0) <= data_host_.extent(0),
                             "left index shrinks");
    PARTHENON_REQUIRE_THROWS(shape_host_(2) <= data_host_.extent(1),
                             "right index shrinks");

    core_data_host_t new_data_host(ViewOfViewAlloc<HostMemSpace>("tensor core host"),
                                   shape_host_(0), shape_host_(2));
    core_data_device_t new_data_device(ViewOfViewAlloc("tensor core device"), shape_host_(0),
                                       shape_host_(2));
    
    // we have to use a mirror to copy to device again
    auto mirror = Kokkos::create_mirror_view(new_data_device);

    for (std::size_t iL = 0; iL < shape_host_(0); iL++) {
      for (std::size_t iR = 0; iR < shape_host_(2); iR++) {
        new_data_host(iL, iR) = data_host_(iL, iR);
        mirror(iL, iR) = new_data_host(iL, iR);
      }
    }
    // this deep copy doesn't work on device because the types are different. use a mirror instead
    // Kokkos::deep_copy(new_data_device, new_data_host);
    Kokkos::deep_copy(new_data_device, mirror);

    data_host_ = new_data_host;
    data_device_ = new_data_device;
  }

 private:
  shape_t shape_;
  shape_host_t shape_host_;
  core_data_host_t data_host_;
  core_data_device_t data_device_;
}; // Class TensorCoreHost
using cores_device_t = Kokkos::View<TensorCoreDevice *, LayoutWrapper, DevMemSpace>;
// TODO(JMM): Should this be a view? Or a std::vector? Any reason to use Kokkos here?
using cores_host_t = Kokkos::View<TensorCoreHost *, LayoutWrapper, HostMemSpace>;

class TensorTrain {
 public:
  // most likely for testing purposes
  TensorTrain(const std::string &name, const std::vector<TensorCoreHost> &cores)
      : label_(name) {
    for (std::size_t i = 0; i < cores.size() - 1; ++i) {
      PARTHENON_REQUIRE_THROWS(cores[i].GetRightRank() == cores[i + 1].GetLeftRank(),
                               "Ranks agree");
    }
    // Host only view-of-views, owns the host only tensor cores
    cores_host_ =
        cores_host_t(ViewOfViewAlloc<HostMemSpace>(name + " host"), cores.size());
    // Device only view-of-views. The device context for cores is
    // non-owning, so no need for SequentialHostInit
    cores_device_ = cores_device_t(ViewOfViewAlloc(name + " device"), cores.size());
    auto cores_device_hm = Kokkos::create_mirror_view(cores_device_);
    for (std::size_t i = 0; i < cores.size(); ++i) {
      cores_host_(i) = cores[i];
      cores_device_hm(i) = cores_host_(i).GetOnDevice();
    }
    Kokkos::deep_copy(cores_device_, cores_device_hm);
  }

  // get number of cores in tensor train
  KOKKOS_INLINE_FUNCTION
  std::size_t GetNumCores() const { return cores_device_.size(); }

  // get left rank for tensor core core_index
  std::size_t GetLeftRank(const int core_index) const {
    return cores_host_(core_index).GetLeftRank();
  }

  // get right rank for tensor core core_index
  std::size_t GetRightRank(const int core_index) const {
    return cores_host_(core_index).GetRightRank();
  }

  // get physical index size rank for tensor core core_index
  std::size_t GetPhysicalIndexSize(const int core_index) const {
    return cores_host_(core_index).GetPhysicalIndexSize();
  }

  // get largest left rank
  KOKKOS_INLINE_FUNCTION
  std::size_t GetMaximumLeftRank() const {
    std::size_t max_left_rank = 0;
    for (int i = 0; i < GetNumCores(); i++) {
      max_left_rank = std::max(max_left_rank, GetLeftRank(i));
    }
    return max_left_rank;
  }

  // get largest right rank
  KOKKOS_INLINE_FUNCTION
  std::size_t GetMaximumRightRank() const {
    std::size_t max_right_rank = 0;
    for (int i = 0; i < GetNumCores(); i++) {
      max_right_rank = std::max(max_right_rank, GetRightRank(i));
    }
    return max_right_rank;
  }

  // get largest physical index size
  KOKKOS_INLINE_FUNCTION
  std::size_t GetMaximumPhysicalIndexSize() const {
    std::size_t max_physical_index_size = 0;
    for (int i = 0; i < GetNumCores(); i++) {
      max_physical_index_size =
          std::max(max_physical_index_size, GetPhysicalIndexSize(i));
    }
    return max_physical_index_size;
  }

  // After resizing a core (on host) following rounding, the device cores are now stale
  // and need to be updated as well
  void SyncDeviceCores() {
    auto mirror = Kokkos::create_mirror_view(cores_device_);

    for (int i = 0; i < GetNumCores(); ++i) {
      mirror(i) = cores_host_(i).GetOnDevice();
    }

    Kokkos::deep_copy(cores_device_, mirror);
  }

  // Evaluates the tensor train and returns the dense array it
  // represents as a Kokkos view. This is mostly for debugging!
  // TODO(JMM): I am giving up on doing this generically. It's not
  // worth it.
  // This assumes that the full rank tensor has 3 dimensions, just for testing.
  // For generality, we would need to support arbitrary dimensionality of
  // parthenon arrays or a clever indexer function with arbitrary dimension
  ParArrayND<Real> ToDenseArray3D() const {
    // Jump through a lot of stupid hoops to generate a ParArrayND
    // TODO(JMM): Only works for howevery many dims ParArrayND supports
    PARTHENON_REQUIRE_THROWS(
        cores_host_.size() <= 3,
        "The dense object must have less dimensions than the output array");
    ParArrayND<Real> out(
        "dense version of " + label_, cores_host_[0].GetPhysicalIndexSize(),
        cores_host_[1].GetPhysicalIndexSize(), cores_host_[2].GetPhysicalIndexSize());
    par_for(
        PARTHENON_AUTO_LABEL, 0, cores_host_[0].GetPhysicalIndexSize() - 1, 0,
        cores_host_[1].GetPhysicalIndexSize() - 1, 0,
        cores_host_[2].GetPhysicalIndexSize() - 1,
        KOKKOS_CLASS_LAMBDA(const int c0, const int c1, const int c2) {
          out(c0, c1, c2) = 0;
          for (std::size_t r01 = 0; r01 < cores_device_(0).GetRightRank(); ++r01) {
            for (std::size_t r12 = 0; r12 < cores_device_(1).GetRightRank(); ++r12) {
              out(c0, c1, c2) += cores_device_(0)(0, c0, r01) *
                                 cores_device_(1)(r01, c1, r12) *
                                 cores_device_(2)(r12, c2, 0);
            }
          }
        });
    return out;
  }

  auto label() const { return label_; }

  // take two tensor trains and a scalar, returning a new tensor object Z = aX + Y
  friend TensorTrain aXPlusY(pool_map_t &pool_map, const Real a, const TensorTrain &X,
                             const TensorTrain &Y);

  void SetOnes() {
    // zero initialize
    auto cores = cores_device_;
    par_for(
        PARTHENON_AUTO_LABEL, 0, GetNumCores() - 1, KOKKOS_LAMBDA(const int i) {
          for (int iL = 0; iL < cores(i).GetLeftRank(); iL++) {
            for (int iR = 0; iR < cores(i).GetRightRank(); iR++) {
              for (int ic = 0; ic < cores(i).GetPhysicalIndexSize(); ic++) {
                cores(i)(iL, ic, iR) = 1.;
              }
            }
          }
        });
  }

  void SetCoreEntry(const int core, const int rL, const int i, const int rR,
                    const Real value) {
    auto cores = cores_device_;
    cores(core)(rL, i, rR) = value;
  }

  // Gram-SVD TT rounding with tolerance eps. Reduces TT ranks while
  // preserving the tensor up to Frobenius error eps.
  void GramSVDRound(const Real eps);

  // TODO(SWJ): remove this and make the things that need it able to
  // access private scope somehow
  // SWJ: I think this can be removed now.
  // const cores_device_t &cores_device() const { return cores_device_; }

  KOKKOS_INLINE_FUNCTION
  void CalculateRightGramMatrices(GramSVDStorage &GS,
                                  const parthenon::team_mbr_t &member);

  KOKKOS_INLINE_FUNCTION
  void CalculateLeftGramMatrices(GramSVDStorage &GS, const parthenon::team_mbr_t &member);
  KOKKOS_INLINE_FUNCTION
  void CalculateGramSVD(const int n, parthenon::team_mbr_t member, GramSVDStorage &GS);

  // examine SVD's singular values for core n and return:
  // keep: flag should we keep (1) or discard (0) this singular values
  // gamma_map: compactified map to retained singular values
  // integer return: number of retained singular values
  KOKKOS_INLINE_FUNCTION
  int SelectSingularModes(const int n, GramSVDStorage &GS, const Real eps) {

    auto cores = this->cores_device_;
    const std::size_t Rn = cores(n).GetRightRank();

    for (int i = 0; i < Rn; i++) {
      GS.KeepFlags()(i) = GS.ModeMap()(i) = -1;
    };

    // find maximum singular value
    Real sigmax{-1.e30};
    int sigmax_loc;
    for (int i = 0; i < Rn; i++) {
      sigmax = std::max(sigmax, std::abs(GS.SVDS()(i)));
      sigmax_loc = i;
    };

    // flag which singular values we should keep
    const Real sigmaxeps = sigmax * eps;
    int Rn_new{0};
    // Singular values from Luke's square SVD routine not guaranteed positive
    for (int i = 0; i < Rn; i++) {
      if (std::abs(GS.SVDS()(i)) > sigmaxeps) {
        GS.KeepFlags()(i) = 1;
        GS.ModeMap()(Rn_new) = i;
        Rn_new++;
      }
    };

    return Rn_new;
  }

  KOKKOS_INLINE_FUNCTION
  void UpdateCoreIndexSpaces(const int n, const int Rn_new, parthenon::team_mbr_t tm,
                             GramSVDStorage &GS);

 private:
  std::string label_;
  cores_host_t cores_host_;
  cores_device_t cores_device_;
}; // class TensorTrain

} // namespace tensors
} // namespace parthenon

#endif // TENSORS_TENSORS_HPP_
