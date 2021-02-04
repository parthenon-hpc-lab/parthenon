//========================================================================================
// (C) (or copyright) 2021. Triad National Security, LLC. All rights reserved.
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
#ifndef INTERFACE_SWARM_BOUNDARIES_HPP_
#define INTERFACE_SWARM_BOUNDARIES_HPP_

#include "swarm_device_context.hpp"

namespace parthenon {

class ParticleBound {
 public:
  KOKKOS_INLINE_FUNCTION virtual void Apply(const int n, double &x, double &y, double &z,
                                                const SwarmDeviceContext &context) = 0;
};

class ParticleBoundIX1Periodic : public ParticleBound {
 public:
  KOKKOS_INLINE_FUNCTION void Apply(const int n, double &x, double &y, double &z,
                                        const SwarmDeviceContext &swarm_d) override {
    if (x < swarm_d.x_min_global_) {
      x = swarm_d.x_max_global_ - (swarm_d.x_min_global_ - x);
    }
  }
};

class ParticleBoundIX1Outflow : public ParticleBound {
 public:
  KOKKOS_INLINE_FUNCTION void Apply(const int n, double &x, double &y, double &z,
                                        const SwarmDeviceContext &swarm_d) override {
    swarm_d.MarkParticleForRemoval(n);
  }
};

class ParticleBoundIX1Reflect : public ParticleBound {
 public:
  KOKKOS_INLINE_FUNCTION void Apply(const int n, double &x, double &y, double &z,
                                        const SwarmDeviceContext &swarm_d) override {
    if (x < swarm_d.x_min_global_) {
      x = swarm_d.x_min_global_ + (swarm_d.x_min_global_ - x);
    }
  }
};

class ParticleBoundOX1Periodic : public ParticleBound {
 public:
  KOKKOS_INLINE_FUNCTION void Apply(const int n, double &x, double &y, double &z,
                                        const SwarmDeviceContext &swarm_d) override {
    if (x > swarm_d.x_max_global_) {
      x = swarm_d.x_min_global_ + (x - swarm_d.x_max_global_);
    }
  }
};

class ParticleBoundOX1Outflow : public ParticleBound {
 public:
  KOKKOS_INLINE_FUNCTION void Apply(const int n, double &x, double &y, double &z,
                                        const SwarmDeviceContext &swarm_d) override {
//                                          printf("Marking particle %i for removal!\n", n);
    swarm_d.MarkParticleForRemoval(n);
  }
};

class ParticleBoundOX1Reflect : public ParticleBound {
 public:
  KOKKOS_INLINE_FUNCTION void Apply(const int n, double &x, double &y, double &z,
                                        const SwarmDeviceContext &swarm_d) override {
    if (x > swarm_d.x_max_global_) {
      x = swarm_d.x_max_global_ - (x - swarm_d.x_max_global_);
    }
  }
};

} // namespace parthenon

#endif // INTERFACE_SWARM_BOUNDARIES_HPP_
