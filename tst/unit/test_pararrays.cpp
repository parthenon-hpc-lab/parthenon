//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2020-2022 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020-2022. Triad National Security, LLC. All rights reserved.
//
// This program was produced under U.S. Government contract 89233218CNA000001
// for Los Alamos National Laboratory (LANL), which is operated by Triad
// National Security, LLC for the U.S. Department of Energy/National Nuclear
// Security Administration. All rights in the program are reserved by Triad
// National Security, LLC, and the U.S. Department of Energy/National Nuclear
// Security Administration. The Government is granted for itself and others
// acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license
// in this material to reproduce, prepare derivative works, distribute copies to
// the public, perform publicly and display publicly, and to permit others to do
// so.
//========================================================================================

#include <cmath>
#include <iostream>
#include <random>
#include <string>

#include <catch2/catch.hpp>

#include "kokkos_abstraction.hpp"
#include "parthenon_arrays.hpp"

using parthenon::DevExecSpace;
using parthenon::DevMemSpace;
using parthenon::ParArray3D;
using parthenon::ParArrayND;
using Real = double;

using policy6d = Kokkos::MDRangePolicy<Kokkos::Rank<6>>;
using policy4d = Kokkos::MDRangePolicy<Kokkos::Rank<4>>;
using policy3d = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
using policy2d = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;

constexpr int NG = 1; // six-point stencil requires one ghost zone
constexpr int N = 32 + 2 * NG;
constexpr int NT = 100;
constexpr int NARRAYS = 64;
constexpr int NS = 2;

#if defined(KOKKOS_ENABLE_CUDA)
using UVMSpace = Kokkos::CudaUVMSpace;
#elif defined(KOKKOS_ENABLE_HIP)
using UVMSpace = Kokkos::Experimental::HIPHostPinnedSpace;
#else // all on host
using UVMSpace = DevMemSpace;
#endif

KOKKOS_INLINE_FUNCTION Real coord(const int i, const int n) {
  const Real dx = 2.0 / (n - 1.0);
  return -1.0 + dx * i;
}

KOKKOS_INLINE_FUNCTION Real gaussian(const int iz, const int iy, const int ix,
                                     const int nz, const int ny, const int nx) {
  const Real x = coord(ix, nx);
  const Real y = coord(iy, ny);
  const Real z = coord(iz, nz);
  const Real r2 = x * x + y * y + z * z;
  return exp(-r2);
}

KOKKOS_INLINE_FUNCTION Real gaussian(const int iz, const int iy, const int ix) {
  return gaussian(iz, iy, ix, N, N, N);
}

template <typename T>
KOKKOS_FORCEINLINE_FUNCTION void stencil(T &l, T &r, const int k, const int j,
                                         const int i) {
  // clang-format off
  l(k,j,i) = (1./6.)*( r(k-1, j,   i)   + r(k+1, j,   i)
                      +r(k,   j-1, i)   + r(k,   j+1, i)
                      +r(k,   j,   i-1) + r(k,   j,   i+1));
  // clang-format on
}

template <class T>
void profile_wrapper_3d(T loop_pattern) {
  auto exec_space = DevExecSpace();
  Kokkos::Timer timer;

  ParArray3D<Real> raw0("raw", N, N, N);
  ParArrayND<Real> nda0("ND", N, N, N);
  auto xtra0 = nda0.Get<3>();

  ParArray3D<Real> raw1("raw", N, N, N);
  ParArrayND<Real> nda1("ND", N, N, N);
  auto xtra1 = nda1.Get<3>();

  parthenon::par_for(
      loop_pattern, "initial data", exec_space, 0, N - 1, 0, N - 1, 0, N - 1,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        Real f = gaussian(k, j, i);
        raw0(k, j, i) = f;
        nda0(k, j, i) = f;
      });
  Kokkos::fence();
  timer.reset();
  for (int it = 0; it < NT; it++) {
    parthenon::par_for(
        loop_pattern, "main loop", exec_space, NG, N - 1 - NG, NG, N - 1 - NG, NG,
        N - 1 - NG, KOKKOS_LAMBDA(const int k, const int j, const int i) {
          stencil(raw1, raw0, k, j, i);
        });
    parthenon::par_for(
        loop_pattern, "main loop", exec_space, NG, N - 1 - NG, NG, N - 1 - NG, NG,
        N - 1 - NG, KOKKOS_LAMBDA(const int k, const int j, const int i) {
          stencil(raw0, raw1, k, j, i);
        });
  }
  Kokkos::fence();
  auto time_raw = timer.seconds();
  timer.reset();
  for (int it = 0; it < NT; it++) {
    parthenon::par_for(
        loop_pattern, "main loop", exec_space, NG, N - 1 - NG, NG, N - 1 - NG, NG,
        N - 1 - NG, KOKKOS_LAMBDA(const int k, const int j, const int i) {
          stencil(nda1, nda0, k, j, i);
        });
    parthenon::par_for(
        loop_pattern, "main loop", exec_space, NG, N - 1 - NG, NG, N - 1 - NG, NG,
        N - 1 - NG, KOKKOS_LAMBDA(const int k, const int j, const int i) {
          stencil(nda0, nda1, k, j, i);
        });
  }
  Kokkos::fence();
  auto time_ND_arrays = timer.seconds();

  timer.reset();
  for (int it = 0; it < NT; it++) {
    parthenon::par_for(
        loop_pattern, "main loop", exec_space, NG, N - 1 - NG, NG, N - 1 - NG, NG,
        N - 1 - NG, KOKKOS_LAMBDA(const int k, const int j, const int i) {
          stencil(xtra1, xtra0, k, j, i);
        });
    parthenon::par_for(
        loop_pattern, "main loop", exec_space, NG, N - 1 - NG, NG, N - 1 - NG, NG,
        N - 1 - NG, KOKKOS_LAMBDA(const int k, const int j, const int i) {
          stencil(xtra0, xtra1, k, j, i);
        });
  }
  Kokkos::fence();
  auto time_extracted = timer.seconds();

  std::cout << "Times for stencil test:\n"
            << "\traw views   = " << time_raw << " s\n"
            << "\tND arrays   = " << time_ND_arrays << " s\n"
            << "\textracted   = " << time_extracted << " s\n"
            << std::endl;
}

TEST_CASE("ParArrayND", "[ParArrayND][Kokkos]") {
  GIVEN("A ParArrayND allocated with no label") {
    ParArrayND<Real> a(PARARRAY_TEMP, 5, 4, 3, 2);
    THEN("The label is the correct default") {
      REQUIRE(a.label().find("ParArrayND") != std::string::npos);
    }
  }

  GIVEN("A ParArrayND with some numbers and its Kokkos host mirror copy") {
    ParArrayND<Real> a("GetMirrorAndCopy", 5, 4, 3);

    Kokkos::parallel_for(
        policy3d({0, 0, 0}, {5, 4, 3}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          a(k, j, i) = static_cast<Real>(k * j * i);
        });

    auto a_host_raw =
        Kokkos::create_mirror_view(static_cast<decltype(a.Get<3>())::base_t>(a.Get<3>()));
    Kokkos::deep_copy(a_host_raw, static_cast<decltype(a.Get<3>())::base_t>(a.Get<3>()));

    THEN("The GetHostMirrorAndCopy is identical") {
      auto a_host_wrap = a.GetHostMirrorAndCopy();
      bool same = true;
      for (int k = 0; k < 5; k++)
        for (int j = 0; j < 4; j++)
          for (int i = 0; i < 3; i++) {
            if (a_host_raw(k, j, i) != a_host_wrap(k, j, i)) {
              same = false;
            }
          }
      REQUIRE(same);
    }
  }

  GIVEN("A ParArrayND with some dimensions") {
    constexpr int N1 = 2;
    constexpr int N2 = 3;
    constexpr int N3 = 4;
    ParArrayND<Real> a("test", N3, N2, N1);
    WHEN("The ParArray legacy NewParArray method is used") {
      ParArrayND<Real> b;
      b = ParArrayND<Real>(PARARRAY_TEMP, N3, N2, N1);
      THEN("The dimensions are correct") {
        REQUIRE(b.GetDim(3) == N3);
        REQUIRE(b.GetDim(2) == N2);
        REQUIRE(b.GetDim(1) == N1);
        for (int d = 4; d <= 6; d++) {
          REQUIRE(b.GetDim(d) == 1);
        }
      }
    }
    WHEN("We fill it with increasing integers") {
      // auto view = a.Get<3>();
      // auto mirror = Kokkos::create_mirror(view);
      auto mirror = a.GetHostMirror();
      int n = 0;
      int sum_host = 0;
      for (int k = 0; k < N3; k++) {
        for (int j = 0; j < N2; j++) {
          for (int i = 0; i < N1; i++) {
            mirror(k, j, i) = n;
            sum_host += n;
            n++;
          }
        }
      }
      // Kokkos::deep_copy(view,mirror);
      a.DeepCopy(mirror);
      THEN("the sum of the lower three indices is correct") {
        int sum_device = 0;
        Kokkos::parallel_reduce(
            policy3d({0, 0, 0}, {N3, N2, N1}),
            KOKKOS_LAMBDA(const int k, const int j, const int i, int &update) {
              update += a(k, j, i);
            },
            sum_device);
        REQUIRE(sum_host == sum_device);
      }
      THEN("the sum of the lower TWO indices is correct") {
        sum_host = 0;
        n = 0;
        for (int j = 0; j < N2; j++) {
          for (int i = 0; i < N1; i++) {
            sum_host += n;
            n++;
          }
        }
        int sum_device = 0;
        Kokkos::parallel_reduce(
            policy2d({0, 0}, {N2, N1}),
            KOKKOS_LAMBDA(const int j, const int i, int &update) { update += a(j, i); },
            sum_device);
        REQUIRE(sum_host == sum_device);
        AND_THEN("We can get a raw 2d subview and it works the same way.") {
          auto v2d = a.Get<2>();
          sum_device = 0;
          Kokkos::parallel_reduce(
              policy2d({0, 0}, {N2, N1}),
              KOKKOS_LAMBDA(const int j, const int i, int &update) {
                update += v2d(j, i);
              },
              sum_device);
          REQUIRE(sum_host == sum_device);
        }
      }
    }
  }
}

TEST_CASE("ParArrayND with LayoutLeft", "[ParArrayND][Kokkos][LayoutLeft]") {
  GIVEN("A ParArrayND with some dimensions") {
    constexpr int N1 = 2;
    constexpr int N2 = 3;
    constexpr int N3 = 4;
    ParArrayND<Real, Kokkos::LayoutLeft> a("test", N3, N2, N1);
    WHEN("We fill it with increasing integers") {
      // auto view = a.Get<3>();
      // auto mirror = Kokkos::create_mirror(view);
      auto mirror = a.GetHostMirror();
      int n = 0;
      int sum_host = 0;
      for (int k = 0; k < N3; k++) {
        for (int j = 0; j < N2; j++) {
          for (int i = 0; i < N1; i++) {
            mirror(k, j, i) = n;
            sum_host += n;
            n++;
          }
        }
      }
      // Kokkos::deep_copy(view,mirror);
      a.DeepCopy(mirror);
      THEN("the sum of the lower three indices is correct") {
        int sum_device = 0;
        Kokkos::parallel_reduce(
            policy3d({0, 0, 0}, {N3, N2, N1}),
            KOKKOS_LAMBDA(const int k, const int j, const int i, int &update) {
              update += a(k, j, i);
            },
            sum_device);
        REQUIRE(sum_host == sum_device);
      }
    }
  }
}

TEST_CASE("ParArray resizing", "[ParArrayND]") {
  GIVEN("Initial ParArrayNDs of multiple extents") {
    ParArrayND<int> pa1("1D", NS);
    ParArrayND<int> pa2("2D", NS, NS);
    ParArrayND<int> pa3("3D", NS, NS, NS);
    ParArrayND<int> pa4("4D", NS, NS, NS, NS);
    ParArrayND<int> pa5("5D", NS, NS, NS, NS, NS);
    ParArrayND<int> pa6("6D", NS, NS, NS, NS, NS, NS);
    Kokkos::parallel_for(
        policy6d({0, 0, 0, 0, 0, 0}, {NS, NS, NS, NS, NS, NS}),
        KOKKOS_LAMBDA(const int n, const int m, const int l, const int k, const int j,
                      const int i) {
          pa1(i) = 1;
          pa2(j, i) = 1;
          pa3(k, j, i) = 1;
          pa4(l, k, j, i) = 1;
          pa5(m, l, k, j, i) = 1;
          pa6(n, m, l, k, j, i) = 1;
        });
    pa1.Resize(NS + 1);
    pa2.Resize(NS + 1, NS + 1);
    pa3.Resize(NS + 1, NS + 1, NS + 1);
    pa4.Resize(NS + 1, NS + 1, NS + 1, NS + 1);
    pa5.Resize(NS + 1, NS + 1, NS + 1, NS + 1, NS + 1);
    pa6.Resize(NS + 1, NS + 1, NS + 1, NS + 1, NS + 1, NS + 1);
    auto pa1_h = pa1.GetHostMirrorAndCopy();
    auto pa2_h = pa2.GetHostMirrorAndCopy();
    auto pa3_h = pa3.GetHostMirrorAndCopy();
    auto pa4_h = pa4.GetHostMirrorAndCopy();
    auto pa5_h = pa5.GetHostMirrorAndCopy();
    auto pa6_h = pa6.GetHostMirrorAndCopy();
    THEN("Resizing them should work and retain data in the common extent") {
      REQUIRE(pa1_h(NS - 1) == 1);
      REQUIRE(pa2_h(NS - 1, NS - 1) == 1);
      REQUIRE(pa3_h(NS - 1, NS - 1, NS - 1) == 1);
      REQUIRE(pa4_h(NS - 1, NS - 1, NS - 1, NS - 1) == 1);
      REQUIRE(pa5_h(NS - 1, NS - 1, NS - 1, NS - 1, NS - 1) == 1);
      REQUIRE(pa6_h(NS - 1, NS - 1, NS - 1, NS - 1, NS - 1, NS - 1) == 1);
    }
    pa1.Resize(1);
    pa2.Resize(1, 2);
    pa3.Resize(1, 2, 3);
    pa4.Resize(1, 2, 3, 4);
    pa5.Resize(1, 2, 3, 4, 5);
    pa6.Resize(1, 2, 3, 4, 5, 6);
    THEN("Resized arrays should have the correct dimensions") {
      REQUIRE(pa1.GetDim(1) == 1);
      REQUIRE(pa2.GetDim(1) == 2);
      REQUIRE(pa2.GetDim(2) == 1);
      REQUIRE(pa3.GetDim(1) == 3);
      REQUIRE(pa3.GetDim(2) == 2);
      REQUIRE(pa3.GetDim(3) == 1);
      REQUIRE(pa4.GetDim(1) == 4);
      REQUIRE(pa4.GetDim(2) == 3);
      REQUIRE(pa4.GetDim(3) == 2);
      REQUIRE(pa4.GetDim(4) == 1);
      REQUIRE(pa5.GetDim(1) == 5);
      REQUIRE(pa5.GetDim(2) == 4);
      REQUIRE(pa5.GetDim(3) == 3);
      REQUIRE(pa5.GetDim(4) == 2);
      REQUIRE(pa5.GetDim(5) == 1);
      REQUIRE(pa6.GetDim(1) == 6);
      REQUIRE(pa6.GetDim(2) == 5);
      REQUIRE(pa6.GetDim(3) == 4);
      REQUIRE(pa6.GetDim(4) == 3);
      REQUIRE(pa6.GetDim(5) == 2);
      REQUIRE(pa6.GetDim(6) == 1);
    }
  }
}

struct state_t : public parthenon::empty_state_t {
  KOKKOS_INLINE_FUNCTION
  state_t() : state_val_(0.0) {}
  KOKKOS_INLINE_FUNCTION
  explicit state_t(double val) : state_val_(val) {}
  KOKKOS_INLINE_FUNCTION
  double val() { return state_val_; }

 private:
  double state_val_;
};

TEST_CASE("ParArray access", "[ParArrayND]") {
  using arr4d_t = parthenon::ParArray4D<double>;
  using arr3d_t = parthenon::ParArray3D<double>;
  GIVEN("A ParArray4D partially loaded with ones based on an enum idx") {
    arr4d_t pa4("4D", 4, N, N, N);
    enum { a = 0, b = 1, c = 2, d = 3 };
    enum class IdxEnum : int { a = 0, b = 1, c = 2, d = 3 };

    arr3d_t pa3 = pa4.Get(IdxEnum::b);

    Kokkos::parallel_for(
        policy3d({0, 0, 0}, {N, N, N}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          pa4(b, k, j, i) = 1.0;
          pa4(IdxEnum::b, k, j, i) = 1.0;
        });

    THEN("The 3D ParArray should be all ones") {
      auto pa3_h = pa3.GetHostMirrorAndCopy();
      for (int k = 0; k < N; ++k)
        for (int j = 0; j < N; ++j)
          for (int i = 0; i < N; ++i) {
            REQUIRE(pa3_h(k, j, i) == 1.0);
          }
    }
  }
}

TEST_CASE("ParArray state", "[ParArrayND]") {
  using arr4d_t = parthenon::ParArray4D<double, state_t>;
  using arr3d_t = parthenon::ParArray3D<double, state_t>;
  GIVEN("A ParArray4D with some associated state and a ParArray3D gotten from it") {
    const double test_value = 5.0;
    state_t state(test_value);

    arr4d_t pa4("4D", state, 4, N, N, N);
    arr3d_t pa3 = pa4.Get(1);

    THEN("The lower dimensional ParArray should have the same state.") {
      REQUIRE(pa3.val() == test_value);
    }
  }

  GIVEN("A ParArray4D and another ParArray4D copied from it") {
    const double test_value = 5.0;
    state_t state(test_value);

    arr4d_t pa4("4D", state, 4, N, N, N);

    arr4d_t pa4_copy = pa4;

    THEN("They point to the same resource") { REQUIRE(UseSameResource(pa4, pa4_copy)); }
  }

  GIVEN("A state type that inherits from empty_state_t") {
    arr3d_t pa3("3D", state_t(5.0), N, N, N);
    THEN("We should be able to copy into a stateless ParArray") {
      ParArray3D<double> pa3_stateless = pa3;
    }
  }

  GIVEN("An array of ParArrays filled with the values contained in their state") {
    parthenon::ParArray1D<arr3d_t> pack("test pack", NS);
    auto pack_h = Kokkos::create_mirror_view(pack);

    for (int b = 0; b < NS; ++b) {
      state_t state(static_cast<double>(b));
      pack_h(b) = arr3d_t("3D", state, N, N, N);
    }
    Kokkos::deep_copy(pack, pack_h);

    Kokkos::parallel_for(
        policy4d({0, 0, 0, 0}, {NS, N, N, N}),
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          pack(b)(k, j, i) = pack(b).val();
        });

    THEN("The arrays should contain the value contained in the state") {
      for (int b = 0; b < NS; ++b) {
        auto pa3_h = pack_h(b).GetHostMirrorAndCopy();

        for (int k = 0; k < N; ++k)
          for (int j = 0; j < N; ++j)
            for (int i = 0; i < N; ++i) {
              REQUIRE(pa3_h(k, j, i) == static_cast<double>(b));
            }
      }
    }
  }
}

// clang-format gets confused by the #ifndef inside the TEST_CASE
// clang-format off
TEST_CASE("Time simple stencil operations", "[ParArrayND][performance]") {
  SECTION("1d range") {
    std::cout << "1d range:" << std::endl;
    profile_wrapper_3d(parthenon::loop_pattern_flatrange_tag);
  }

  // skip this output for now, since times are comparable
  // SECTION("md range") {
  //   std::cout << "md range:" << std::endl;
  //   profile_wrapper_3d(parthenon::loop_pattern_mdrange_tag);
  // }
  // SECTION("tpttr") {
  //   std::cout << "tpttr range:" << std::endl;
  //   profile_wrapper_3d(parthenon::loop_pattern_tpttr_tag);
  // }
  // SECTION("tpttrtvr") {
  //   std::cout << "tpttrvr range:" << std::endl;
  //   profile_wrapper_3d(parthenon::loop_pattern_tpttrtvr_tag);
  // }

#if !(defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP))
  // SECTION("tptvr") {
  //   std::cout << "tptvr range:" << std::endl;
  //   profile_wrapper_3d(parthenon::loop_pattern_tptvr_tag);
  // }

  SECTION("simdfor") {
    std::cout << "simd range:" << std::endl;
    profile_wrapper_3d(parthenon::loop_pattern_simdfor_tag);
  }
#endif // !KOKKOS_ENABLE_CUDA
}
// clang-format on

TEST_CASE("Check registry pressure", "[ParArrayND][performance]") {
  auto exec_space = DevExecSpace();
  Kokkos::Timer timer;

  // https://en.cppreference.com/w/cpp/numeric/random/uniform_real_distribution
  std::random_device rd;  // Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<Real> dis(-1.0, 1.0);

  // view of views. See:
  // https://github.com/kokkos/kokkos/wiki/View#6232-whats-the-problem-with-a-view-of-views
  using view_3d_t =
      Kokkos::View<Real ***, parthenon::LayoutWrapper, parthenon::DevMemSpace>;
  using arrays_t = Kokkos::View<ParArrayND<Real> *, UVMSpace>;
  using views_t = Kokkos::View<view_3d_t *, UVMSpace>;
  using device_view_t = parthenon::device_view_t<Real>;
  arrays_t arrays(Kokkos::view_alloc(std::string("arrays"), Kokkos::WithoutInitializing),
                  NARRAYS);
  views_t views(Kokkos::view_alloc(std::string("views"), Kokkos::WithoutInitializing),
                NARRAYS);
  for (int n = 0; n < NARRAYS; n++) {
    std::string label = std::string("array ") + std::to_string(n);
    new (&arrays[n]) ParArrayND<Real>(device_view_t(
        Kokkos::view_alloc(label, Kokkos::WithoutInitializing), 1, 1, 1, N, N, N));
    label = std::string("view ") + std::to_string(n);
    new (&views[n])
        view_3d_t(Kokkos::view_alloc(label, Kokkos::WithoutInitializing), N, N, N);
    auto a_h = arrays(n).GetHostMirror();
    auto v_h = Kokkos::create_mirror_view(views(n));
    for (int k = 0; k < N; k++) {
      for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {
          a_h(k, j, i) = dis(gen);
          v_h(k, j, i) = dis(gen);
        }
      }
    }
    Kokkos::deep_copy(views(n), v_h);
    arrays(n).DeepCopy(a_h);
  }
  // perform a compute intensive, non-stencil, task and see
  // performance
  Kokkos::fence();
  timer.reset();
  parthenon::par_for(
      parthenon::loop_pattern_flatrange_tag, "compute intensive task for raw views",
      exec_space, 0, N - 1, 0, N - 1, 0, N - 1,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        for (int n = 0; n < NARRAYS; n++) {
          views(n)(k, j, i) = exp(views(n)(k, j, i));
        }
      });
  Kokkos::fence();
  auto time_views = timer.seconds();
  timer.reset();
  parthenon::par_for(
      parthenon::loop_pattern_flatrange_tag, "compute intensive task for ParArrayND",
      exec_space, 0, N - 1, 0, N - 1, 0, N - 1,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        for (int n = 0; n < NARRAYS; n++) {
          arrays(n)(k, j, i) = exp(arrays(n)(k, j, i));
        }
      });
  Kokkos::fence();
  auto time_arrays = timer.seconds();
  std::cout << "Times for register pressure test:\n"
            << "\traw views   = " << time_views << " s\n"
            << "\tND arrays   = " << time_arrays << " s\n"
            << std::endl;
}

template <typename Array>
KOKKOS_FORCEINLINE_FUNCTION void
many_array_kernel(const Array &arr0, const Array &arr1, const Array &arr2,
                  const Array &arr3, const Array &arr4, const Array &arr5,
                  const Array &arr6, const Array &arr7, const Array &arr8,
                  const Array &arr9, const Array &arr_out, const int k, const int j,
                  const int i) {
  for (int rep = 0; rep < 2; rep++) {
    Real tmp_array[10];
    tmp_array[0] = arr0(k, j, i);
    tmp_array[1] = arr1(k, j, i);
    tmp_array[2] = arr2(k, j, i);
    tmp_array[3] = arr3(k, j, i);
    tmp_array[4] = arr4(k, j, i);
    tmp_array[5] = arr5(k, j, i);
    tmp_array[6] = arr6(k, j, i);
    tmp_array[7] = arr7(k, j, i);
    tmp_array[8] = arr8(k, j, i);
    tmp_array[9] = arr9(k, j, i);
    arr_out(k, j, i) = tmp_array[0] + tmp_array[1] + tmp_array[2] + tmp_array[3] +
                       tmp_array[4] + tmp_array[5] + tmp_array[6] + tmp_array[7] +
                       tmp_array[8] + tmp_array[9];
    arr_out(k, j, i) *= tmp_array[0] * tmp_array[1] * tmp_array[2] * tmp_array[3] *
                        tmp_array[4] * tmp_array[5] * tmp_array[6] * tmp_array[7] *
                        tmp_array[8] * tmp_array[9];
  }
}

TEST_CASE("Check many arrays", "[ParArrayND][performance]") {
  auto exec_space = DevExecSpace();
  Kokkos::Timer timer;

  // https://en.cppreference.com/w/cpp/numeric/random/uniform_real_distribution
  std::random_device rd;  // Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<Real> dis(-1.0, 1.0);

  // view of views. See:
  // https://github.com/kokkos/kokkos/wiki/View
  ParArray3D<Real> raw0("raw0", N, N, N), raw1("raw1", N, N, N), raw2("raw2", N, N, N),
      raw3("raw3", N, N, N), raw4("raw4", N, N, N), raw5("raw5", N, N, N),
      raw6("raw6", N, N, N), raw7("raw7", N, N, N), raw8("raw8", N, N, N),
      raw9("raw9", N, N, N), raw_out("raw_out", N, N, N);
  ParArrayND<Real> nda0("nda0", N, N, N), nda1("nda1", N, N, N), nda2("nda2", N, N, N),
      nda3("nda3", N, N, N), nda4("nda4", N, N, N), nda5("nda5", N, N, N),
      nda6("nda6", N, N, N), nda7("nda7", N, N, N), nda8("nda8", N, N, N),
      nda9("nda9", N, N, N), nda_out("nda_out", N, N, N);
  auto xtra0 = nda0.Get<3>();
  auto xtra1 = nda1.Get<3>();
  auto xtra2 = nda2.Get<3>();
  auto xtra3 = nda3.Get<3>();
  auto xtra4 = nda4.Get<3>();
  auto xtra5 = nda5.Get<3>();
  auto xtra6 = nda6.Get<3>();
  auto xtra7 = nda7.Get<3>();
  auto xtra8 = nda8.Get<3>();
  auto xtra9 = nda9.Get<3>();
  auto xtra_out = nda_out.Get<3>();

  parthenon::par_for(
      parthenon::loop_pattern_flatrange_tag, "initial data", exec_space, 0, N - 1, 0,
      N - 1, 0, N - 1, KOKKOS_LAMBDA(const int k, const int j, const int i) {
        Real f = gaussian(k, j, i);
        raw0(k, j, i) = f;
        raw1(k, j, i) = f;
        raw2(k, j, i) = f;
        raw3(k, j, i) = f;
        raw4(k, j, i) = f;
        raw5(k, j, i) = f;
        raw6(k, j, i) = f;
        raw7(k, j, i) = f;
        raw8(k, j, i) = f;
        raw9(k, j, i) = f;
        xtra0(k, j, i) = f;
        xtra1(k, j, i) = f;
        xtra2(k, j, i) = f;
        xtra3(k, j, i) = f;
        xtra4(k, j, i) = f;
        xtra5(k, j, i) = f;
        xtra6(k, j, i) = f;
        xtra7(k, j, i) = f;
        xtra8(k, j, i) = f;
        xtra9(k, j, i) = f;
      });

  // Perform a kernel with many arrays/views, to test high memory and register usage
  const int nruns = 10;
  // Time Raw Views
  // Warmup first
  for (int run = 0; run < nruns; run++) {
    parthenon::par_for(
        parthenon::loop_pattern_flatrange_tag, "11 Views in one kernel", exec_space, 0,
        N - 1, 0, N - 1, 0, N - 1, KOKKOS_LAMBDA(const int k, const int j, const int i) {
          many_array_kernel(raw0, raw1, raw2, raw3, raw4, raw5, raw6, raw7, raw8, raw9,
                            raw_out, k, j, i);
        });
  }
  Kokkos::fence();
  timer.reset();
  for (int run = 0; run < nruns; run++) {
    parthenon::par_for(
        parthenon::loop_pattern_flatrange_tag, "11 Views in one kernel", exec_space, 0,
        N - 1, 0, N - 1, 0, N - 1, KOKKOS_LAMBDA(const int k, const int j, const int i) {
          many_array_kernel(raw0, raw1, raw2, raw3, raw4, raw5, raw6, raw7, raw8, raw9,
                            raw_out, k, j, i);
        });
  }
  Kokkos::fence();
  auto time_views = timer.seconds();

  // Time ParArrayND
  // Warmup first
  for (int run = 0; run < nruns; run++) {
    parthenon::par_for(
        parthenon::loop_pattern_flatrange_tag, "11 ParArrayNDs in one kernel", exec_space,
        0, N - 1, 0, N - 1, 0, N - 1,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          many_array_kernel(nda0, nda1, nda2, nda3, nda4, nda5, nda6, nda7, nda8, nda9,
                            nda_out, k, j, i);
        });
  }
  Kokkos::fence();
  timer.reset();
  for (int run = 0; run < nruns; run++) {
    parthenon::par_for(
        parthenon::loop_pattern_flatrange_tag, "11 ParArrayNDs in one kernel", exec_space,
        0, N - 1, 0, N - 1, 0, N - 1,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          many_array_kernel(nda0, nda1, nda2, nda3, nda4, nda5, nda6, nda7, nda8, nda9,
                            nda_out, k, j, i);
        });
  }
  Kokkos::fence();
  auto time_arrays = timer.seconds();

  // Time subviews
  // Warmup first
  for (int run = 0; run < nruns; run++) {
    parthenon::par_for(
        parthenon::loop_pattern_flatrange_tag, "11 ParArrayND.Get<3>() in one kernel",
        exec_space, 0, N - 1, 0, N - 1, 0, N - 1,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          many_array_kernel(xtra0, xtra1, xtra2, xtra3, xtra4, xtra5, xtra6, xtra7, xtra8,
                            xtra9, xtra_out, k, j, i);
        });
  }
  Kokkos::fence();
  timer.reset();
  for (int run = 0; run < nruns; run++) {
    parthenon::par_for(
        parthenon::loop_pattern_flatrange_tag, "11 ParArrayND.Get<3>() in one kernel",
        exec_space, 0, N - 1, 0, N - 1, 0, N - 1,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          many_array_kernel(xtra0, xtra1, xtra2, xtra3, xtra4, xtra5, xtra6, xtra7, xtra8,
                            xtra9, xtra_out, k, j, i);
        });
  }
  Kokkos::fence();
  auto time_subviews = timer.seconds();

  std::cout << "Times for many arrays test:\n"
            << "\traw views   = " << time_views << " s\n"
            << "\tND arrays   = " << time_arrays << " s\n"
            << "\tsub views   = " << time_subviews << " s\n"
            << std::endl;
}
