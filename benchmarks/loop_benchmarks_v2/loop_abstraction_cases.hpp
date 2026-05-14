#pragma once

#include <array>

#include "dataset.hpp"
#include "kernels.hpp"
#include "loop_abstraction.hpp"

namespace plb2 {

template <loop_abstraction::loop_tag LOOP_TAG, loop_abstraction::inner_tag INNER_TAG,
          int SX, int SY, int SZ>
void RunLoopAbstractionCase(const CaseSpec &spec, const Dataset &dataset,
                            const std::array<int, SX> &dx, const std::array<int, SY> &dy,
                            const std::array<int, SZ> &dz,
                            const std::array<double, kMaxNiter> &alpha,
                            const std::array<double, kMaxNiter> &beta);

}  // namespace plb2
