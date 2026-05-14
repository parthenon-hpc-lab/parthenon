#include "loop_abstraction_cases.hpp"

#include <optional>

#include "loop_abstraction_kernel.hpp"

namespace plb2 {

template <loop_abstraction::loop_tag LOOP_TAG, loop_abstraction::inner_tag INNER_TAG,
          int SX, int SY, int SZ>
void RunLoopAbstractionCase(const CaseSpec &spec, const Dataset &dataset,
                            const std::array<int, SX> &dx, const std::array<int, SY> &dy,
                            const std::array<int, SZ> &dz,
                            const std::array<double, kMaxNiter> &alpha,
                            const std::array<double, kMaxNiter> &beta) {
  const std::optional<int> ninner =
      spec.loop.ninner > 0 ? std::optional<int>{spec.loop.ninner} : std::nullopt;
  const auto &problem = dataset.problem;
  RunUnifiedKernelWithLoopAbstraction<LOOP_TAG, INNER_TAG, SX, SY, SZ>(
      dataset.data.in, dataset.data.out, dataset.data.active_counts, problem.nblocks,
      problem.nx_interior, problem.ny_interior, problem.nz_interior, problem.nghost, dx, dy, dz,
      alpha, beta, spec.kernel.niter, ninner);
}

#define PLB2_INSTANTIATE_ABSTRACTION_CASE(LOOP_TAG, INNER_TAG, SX, SY, SZ) \
  template void RunLoopAbstractionCase<loop_abstraction::loop_tag::LOOP_TAG, \
                                       loop_abstraction::inner_tag::INNER_TAG, SX, SY, SZ>( \
      const CaseSpec &, const Dataset &, const std::array<int, SX> &, \
      const std::array<int, SY> &, const std::array<int, SZ> &, \
      const std::array<double, kMaxNiter> &, const std::array<double, kMaxNiter> &)

#define PLB2_INSTANTIATE_ABSTRACTION_STENCILS(LOOP_TAG, INNER_TAG) \
  PLB2_INSTANTIATE_ABSTRACTION_CASE(LOOP_TAG, INNER_TAG, 3, 1, 1); \
  PLB2_INSTANTIATE_ABSTRACTION_CASE(LOOP_TAG, INNER_TAG, 1, 3, 1); \
  PLB2_INSTANTIATE_ABSTRACTION_CASE(LOOP_TAG, INNER_TAG, 1, 1, 3); \
  PLB2_INSTANTIATE_ABSTRACTION_CASE(LOOP_TAG, INNER_TAG, 1, 1, 1)

PLB2_INSTANTIATE_ABSTRACTION_STENCILS(bovi, memory);
PLB2_INSTANTIATE_ABSTRACTION_STENCILS(bovi, logical);
PLB2_INSTANTIATE_ABSTRACTION_STENCILS(boiv, logical);
PLB2_INSTANTIATE_ABSTRACTION_STENCILS(bvoi, memory);
PLB2_INSTANTIATE_ABSTRACTION_STENCILS(bvoi, logical);

#undef PLB2_INSTANTIATE_ABSTRACTION_STENCILS
#undef PLB2_INSTANTIATE_ABSTRACTION_CASE

}  // namespace plb2
