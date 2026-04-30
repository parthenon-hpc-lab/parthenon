//=========================================================================================
// (C) (or copyright) 2026. Triad National Security, LLC. All rights reserved.
//=========================================================================================

#include <Kokkos_Core.hpp>

#include "loop_abstraction.hpp"

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  {
    using namespace plb2; 
    using View5D = Kokkos::View<double *****, Kokkos::LayoutRight>;

    const int n = 2; 
    const int nghost = 1; 
    const int nblocks = 1;
    const int nvar = 2;
    loop_abstraction::index_space_t<loop_abstraction::loop_tag::bvoi,
                                    loop_abstraction::inner_tag::logical> idx_space(nblocks, n, n, n, nghost);
    
    View5D data("data", nblocks, nvar, n, n, n);

    loop_abstraction::outer(idx_space, KOKKOS_LAMBDA(const auto &idx_space, const auto& idx_range, int b){
      for (int v = 0; v < nvar; ++v) {
        auto var = idx_space.GetInnerView(data, b, v);
        loop_abstraction::inner(idx_space, idx_range, [&](auto idx) {
          var(idx) = 1.0;
          auto [k, j, i] = idx_range.GetSpatialIndices(idx);
          printf("  (%i, %i, %i)\n", k, j, i);
        });
      }
    });
    
  }
  Kokkos::finalize();
  return 0;
}
