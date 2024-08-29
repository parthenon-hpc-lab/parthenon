#ifndef _MCCIRC_MCCIRC_HPP_
#define _MCCIRC_MCCIRC_HPP_

#include <memory>
#include "Kokkos_Random.hpp"
#include <parthenon/package.hpp>
#include <interface/swarm_default_names.hpp>

namespace MCCirc {
using namespace parthenon::package::prelude;

typedef Kokkos::Random_XorShift64_Pool<> RNGPool;

struct NumParticles : public parthenon::variable_names::base_t<false> {
  template<class... Ts>
  KOKKOS_INLINE_FUNCTION NumParticles(Ts &&...args)
    : parthenon::variable_names::base_t<false>(std::forward<Ts>(args)...) {}
  static std::string name() { return "particles_per_cell"; }
};
SWARM_VARIABLE(Real, mc, weight);

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);
void ComputeParticleCounts(Mesh *pm);

} // namespace MCCirc

#endif // _MCCIRC_MCCIRC_HPP_
