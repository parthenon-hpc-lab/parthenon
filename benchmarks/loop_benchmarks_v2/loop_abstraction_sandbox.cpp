//=========================================================================================
// (C) (or copyright) 2026. Triad National Security, LLC. All rights reserved.
//=========================================================================================

#include <Kokkos_Core.hpp>
#include <cmath>
#include <cstdio>

#include "loop_abstraction.hpp"

using IS = plb2::loop_abstraction::IndexSpace<
    plb2::loop_abstraction::loop_tag::bovi,
    plb2::loop_abstraction::inner_tag::memory>;

using IR = plb2::loop_abstraction::InnerIndexRange<IS>;

using VW = plb2::loop_abstraction::field_view_t<IS>;

extern "C"
__attribute__((noinline))
void raw_inner_probe(const IR& idx_range,
                     VW& outp,
                     VW& inp) {
  plb2::loop_abstraction::inner(idx_range,
    [&](auto idx) {
      outp(idx) = inp(idx) * 2.01 + outp(idx);
    });
}
namespace {


template <plb2::loop_abstraction::loop_tag LOOP_TAG,
          plb2::loop_abstraction::inner_tag INNER_TAG, class View5D>
void RunKernel(const View5D &input, View5D &output, int nblocks, int nvar, int n,
               int nghost) {
  using namespace plb2;

  loop_abstraction::IndexSpace<LOOP_TAG, INNER_TAG> idx_space(nblocks, n, n, n, nghost);
  loop_abstraction::outer(idx_space, KOKKOS_LAMBDA(const auto &idx_range, int /*b*/) {
    for (int v = 0; v < nvar; ++v) {
      auto in = loop_abstraction::GetView(idx_range, input, v);
      auto out = loop_abstraction::GetView(idx_range, output, v);

      // Just to verify vectorization
      // raw_inner_probe(idx_range, out, in);
      
      loop_abstraction::inner(idx_range, KOKKOS_LAMBDA(auto idx) {
        out(idx) = in(idx) *  2.01 + out(idx);
      });
    }
    
    auto in0 = loop_abstraction::GetView(idx_range, input, 0);
    auto in1 = loop_abstraction::GetView(idx_range, input, 1, {1, 0, 1});
    auto out = loop_abstraction::GetView(idx_range, output, 0);
    loop_abstraction::inner(idx_range, KOKKOS_LAMBDA(auto idx) {
      out(idx) = in0(idx) *  2.01 + in1(idx) * 3.12341 + out(idx);
    });
  });
}

} // namespace

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  {
    using namespace plb2; 
    using View5D = Kokkos::View<double *****, Kokkos::LayoutRight>;

    const int n = 32; 
    const int nghost = 2; 
    const int nblocks = 64;
    const int nvar = 2;
    const int ntot = n + 2 * nghost;
    View5D input("input", nblocks, nvar, ntot, ntot, ntot);
    View5D output("output", nblocks, nvar, ntot, ntot, ntot);

    for (int b = 0; b < nblocks; ++b) {
      for (int v = 0; v < nvar; ++v) {
        for (int k = 0; k < ntot; ++k) {
          for (int j = 0; j < ntot; ++j) {
            for (int i = 0; i < ntot; ++i) {
              input(b, v, k, j, i) = 1.0 + b + 10.01 * v + 100.012 * k + 1000.012 * j + 10000.012 * i;
              output(b, v, k, j, i) = 0.0;
            }
          }
        }
      }
    }

    RunKernel<loop_abstraction::loop_tag::bovi, loop_abstraction::inner_tag::memory>(
      input, output, nblocks, nvar, n, nghost);

    double checksum = 0.0;
    for (int b = 0; b < nblocks; ++b) {
      for (int v = 0; v < nvar; ++v) {
        for (int k = 0; k < ntot; ++k) {
          for (int j = 0; j < ntot; ++j) {
            for (int i = 0; i < ntot; ++i) {
              checksum += output(b, v, k, j, i);
            }
          }
        }
      }
    }

    std::printf("%f\n", checksum);
  }
  Kokkos::finalize();
  return 0;
}
