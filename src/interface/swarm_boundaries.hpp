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
  KOKKOS_INLINE_FUNCTION virtual void Apply(const int n, Real &x, Real &y, Real &z,
                                            const SwarmDeviceContext &context) const {
    PARTHENON_FAIL("You are calling the ParticleBound abstract base class Apply method");
  }
};

class ParticleBoundIX1Periodic : public ParticleBound {
 public:
  KOKKOS_INLINE_FUNCTION void Apply(const int n, Real &x, Real &y, Real &z,
                                    const SwarmDeviceContext &swarm_d) const override {
    if (x < swarm_d.x_min_global_) {
      x = swarm_d.x_max_global_ - (swarm_d.x_min_global_ - x);
    }
  }
};

class ParticleBoundIX1Outflow : public ParticleBound {
 public:
  KOKKOS_INLINE_FUNCTION void Apply(const int n, Real &x, Real &y, Real &z,
                                    const SwarmDeviceContext &swarm_d) const override {
    if (x < swarm_d.x_min_global_) {
      swarm_d.MarkParticleForRemoval(n);
    }
  }
};

class ParticleBoundOX1Periodic : public ParticleBound {
 public:
  KOKKOS_INLINE_FUNCTION void Apply(const int n, Real &x, Real &y, Real &z,
                                    const SwarmDeviceContext &swarm_d) const override {
    if (x > swarm_d.x_max_global_) {
      x = swarm_d.x_min_global_ + (x - swarm_d.x_max_global_);
    }
  }
};

class ParticleBoundOX1Outflow : public ParticleBound {
 public:
  KOKKOS_INLINE_FUNCTION void Apply(const int n, Real &x, Real &y, Real &z,
                                    const SwarmDeviceContext &swarm_d) const override {
    if (x > swarm_d.x_max_global_) {
      swarm_d.MarkParticleForRemoval(n);
    }
  }
};

class ParticleBoundIX2Periodic : public ParticleBound {
 public:
  KOKKOS_INLINE_FUNCTION void Apply(const int n, Real &x, Real &y, Real &z,
                                    const SwarmDeviceContext &swarm_d) const override {
    if (y < swarm_d.y_min_global_) {
      y = swarm_d.y_max_global_ - (swarm_d.y_min_global_ - y);
    }
  }
};

class ParticleBoundIX2Outflow : public ParticleBound {
 public:
  KOKKOS_INLINE_FUNCTION void Apply(const int n, Real &x, Real &y, Real &z,
                                    const SwarmDeviceContext &swarm_d) const override {
    if (y < swarm_d.y_min_global_) {
      swarm_d.MarkParticleForRemoval(n);
    }
  }
};

class ParticleBoundOX2Periodic : public ParticleBound {
 public:
  KOKKOS_INLINE_FUNCTION void Apply(const int n, Real &x, Real &y, Real &z,
                                    const SwarmDeviceContext &swarm_d) const override {
    if (y > swarm_d.y_max_global_) {
      y = swarm_d.y_min_global_ + (y - swarm_d.y_max_global_);
    }
  }
};

class ParticleBoundOX2Outflow : public ParticleBound {
 public:
  KOKKOS_INLINE_FUNCTION void Apply(const int n, Real &x, Real &y, Real &z,
                                    const SwarmDeviceContext &swarm_d) const override {
    if (y > swarm_d.y_max_global_) {
      swarm_d.MarkParticleForRemoval(n);
    }
  }
};

class ParticleBoundIX3Periodic : public ParticleBound {
 public:
  KOKKOS_INLINE_FUNCTION void Apply(const int n, Real &x, Real &y, Real &z,
                                    const SwarmDeviceContext &swarm_d) const override {
    if (z < swarm_d.z_min_global_) {
      z = swarm_d.z_max_global_ - (swarm_d.z_min_global_ - z);
    }
  }
};

class ParticleBoundIX3Outflow : public ParticleBound {
 public:
  KOKKOS_INLINE_FUNCTION void Apply(const int n, Real &x, Real &y, Real &z,
                                    const SwarmDeviceContext &swarm_d) const override {
    if (z < swarm_d.z_min_global_) {
      swarm_d.MarkParticleForRemoval(n);
    }
  }
};

class ParticleBoundOX3Periodic : public ParticleBound {
 public:
  KOKKOS_INLINE_FUNCTION void Apply(const int n, Real &x, Real &y, Real &z,
                                    const SwarmDeviceContext &swarm_d) const override {
    if (z > swarm_d.z_max_global_) {
      z = swarm_d.z_min_global_ + (z - swarm_d.z_max_global_);
    }
  }
};

class ParticleBoundOX3Outflow : public ParticleBound {
 public:
  KOKKOS_INLINE_FUNCTION void Apply(const int n, Real &x, Real &y, Real &z,
                                    const SwarmDeviceContext &swarm_d) const override {
    if (z > swarm_d.z_max_global_) {
      swarm_d.MarkParticleForRemoval(n);
    }
  }
};

} // namespace parthenon

#endif // INTERFACE_SWARM_BOUNDARIES_HPP_
