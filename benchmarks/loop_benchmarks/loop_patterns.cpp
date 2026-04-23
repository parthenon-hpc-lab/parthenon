#include "loop_patterns.hpp"

#include <stdexcept>

#include <Kokkos_Core.hpp>

#include "decomposition.hpp"
#include "kernels.hpp"

namespace plb {

namespace {

using TeamPolicy = Kokkos::TeamPolicy<>;
using TeamMember = TeamPolicy::member_type;
using MDRange4 = Kokkos::MDRangePolicy<Kokkos::Rank<4>>;

RawMemoryIndexer BuildRawMemoryIndexer(const ProblemShape &shape, int inner_length) {
  return RawMemoryIndexer(inner_length, shape.ndim, shape.domain_k, shape.domain_j,
                          shape.domain_i, shape.memory_k, shape.memory_j, shape.memory_i);
}

RawMemoryIndexer BuildTunedIndexer(const ProblemShape &shape, const BenchmarkConfig &config) {
  return BuildRawMemoryIndexer(
      shape, static_cast<int>(DefaultTunedChunkLength(config.ni, config.inner_chunk_length)));
}

ProblemShape BuildProblemShape(const BenchmarkConfig &config) {
  const parthenon::IndexRange block_range{0, config.blocks - 1};
  const int ndim = config.nk > 1 ? 3 : (config.nj > 1 ? 2 : 1);

  ProblemShape shape;
  shape.blocks = config.blocks;
  shape.variables = config.variables;
  shape.nk = config.nk;
  shape.nj = config.nj;
  shape.ni = config.ni;
  shape.ndim = ndim;
  shape.nghost = config.ghost_zones;
  shape.interior_k = {0, config.nk - 1};
  shape.interior_j = {0, config.nj - 1};
  shape.interior_i = {0, config.ni - 1};
  shape.memory_k = {0, config.nk + (ndim > 2 ? 2 * config.ghost_zones : 0) - 1};
  shape.memory_j = {0, config.nj + (ndim > 1 ? 2 * config.ghost_zones : 0) - 1};
  shape.memory_i = {0, config.ni + 2 * config.ghost_zones - 1};
  shape.domain_k = {shape.interior_k.s + (ndim > 2 ? config.ghost_zones : 0),
                    shape.interior_k.e + (ndim > 2 ? config.ghost_zones : 0)};
  shape.domain_j = {shape.interior_j.s + (ndim > 1 ? config.ghost_zones : 0),
                    shape.interior_j.e + (ndim > 1 ? config.ghost_zones : 0)};
  shape.domain_i = {shape.interior_i.s + config.ghost_zones,
                    shape.interior_i.e + config.ghost_zones};
  shape.cell_indexer =
      parthenon::Indexer4D(block_range, shape.domain_k, shape.domain_j, shape.domain_i);
  return shape;
}

void InitializeLoopData(const ProblemShape &shape, LoopData *data) {
  const int nk_mem = shape.memory_k.e - shape.memory_k.s + 1;
  const int nj_mem = shape.memory_j.e - shape.memory_j.s + 1;
  const int ni_mem = shape.memory_i.e - shape.memory_i.s + 1;
  data->in = View5D("in", shape.blocks, shape.variables, nk_mem, nj_mem, ni_mem);
  data->aux = View5D("aux", shape.blocks, shape.variables, nk_mem, nj_mem, ni_mem);
  data->out = View5D("out", shape.blocks, shape.variables, nk_mem, nj_mem, ni_mem);
  data->fx_up = View5D("fx_up", shape.blocks, shape.variables, nk_mem, nj_mem, ni_mem);
  data->fx_lo = View5D("fx_lo", shape.blocks, shape.variables, nk_mem, nj_mem, ni_mem);
  data->fy_up = View5D("fy_up", shape.blocks, shape.variables, nk_mem, nj_mem, ni_mem);
  data->fy_lo = View5D("fy_lo", shape.blocks, shape.variables, nk_mem, nj_mem, ni_mem);
  data->fz_up = View5D("fz_up", shape.blocks, shape.variables, nk_mem, nj_mem, ni_mem);
  data->fz_lo = View5D("fz_lo", shape.blocks, shape.variables, nk_mem, nj_mem, ni_mem);
  data->active_counts = Kokkos::View<int *>("active_counts", shape.blocks);

  const auto in = data->in;
  const auto aux = data->aux;
  const auto out = data->out;
  const auto fx_up = data->fx_up;
  const auto fx_lo = data->fx_lo;
  const auto fy_up = data->fy_up;
  const auto fy_lo = data->fy_lo;
  const auto fz_up = data->fz_up;
  const auto fz_lo = data->fz_lo;

  Kokkos::parallel_for(
      "InitializeData",
      Kokkos::MDRangePolicy<Kokkos::Rank<5>>({0, 0, 0, 0, 0},
                                             {shape.blocks, shape.variables, nk_mem, nj_mem,
                                              ni_mem}),
      KOKKOS_LAMBDA(const int b, const int v, const int k, const int j, const int i) {
        const double seed = static_cast<double>(1 + i + 17 * j + 31 * k + 101 * v + 1009 * b);
        in(b, v, k, j, i) = 0.25 + 0.001 * seed;
        aux(b, v, k, j, i) = 0.75 + 0.002 * seed;
        out(b, v, k, j, i) = 0.0;
        fx_up(b, v, k, j, i) = 0.50 + 0.0010 * seed;
        fx_lo(b, v, k, j, i) = 0.45 + 0.0011 * seed;
        fy_up(b, v, k, j, i) = 0.55 + 0.0012 * seed;
        fy_lo(b, v, k, j, i) = 0.48 + 0.0013 * seed;
        fz_up(b, v, k, j, i) = 0.60 + 0.0014 * seed;
        fz_lo(b, v, k, j, i) = 0.51 + 0.0015 * seed;
      });
  Kokkos::fence();
}

void SetActiveCounts(const BenchmarkConfig &config, const RaggedMetadata &metadata,
                     LoopData *data) {
  auto host = Kokkos::create_mirror_view(data->active_counts);
  for (int block = 0; block < config.blocks; ++block) {
    host(block) = ActiveVariablesForBlock(metadata, config.ragged, block, config.variables);
  }
  Kokkos::deep_copy(data->active_counts, host);
}

KOKKOS_INLINE_FUNCTION
int MemoryJStride(const ProblemShape &shape) { return shape.memory_i.size(); }

KOKKOS_INLINE_FUNCTION
int MemoryKStride(const ProblemShape &shape) { return shape.memory_i.size() * shape.memory_j.size(); }

template <typename Body>
void RunCpuSIMDLoop(const Dataset &dataset, Body body) {
  const auto &shape = dataset.problem;
  auto &data = dataset.data;
  for (int b = 0; b < shape.blocks; ++b) {
    const int nvar = data.active_counts(b);
    for (int v = 0; v < nvar; ++v) {
      for (int km = shape.domain_k.s; km <= shape.domain_k.e; ++km) {
        for (int jm = shape.domain_j.s; jm <= shape.domain_j.e; ++jm) {
#pragma omp simd
          for (int im = shape.domain_i.s; im <= shape.domain_i.e; ++im) {
            data.out(b, v, km, jm, im) = body(data, b, v, km, jm, im);
          }
        }
      }
    }
  }
}

void RunCpuSIMDLight(const Dataset &dataset) {
  RunCpuSIMDLoop(dataset, [](const LoopData &data, const int b, const int v, const int km,
                             const int jm, const int im) {
    return ComputeLightCell(data.in(b, v, km, jm, im), data.aux(b, v, km, jm, im));
  });
}

void RunCpuSIMDFlux(const Dataset &dataset) {
  RunCpuSIMDLoop(dataset, [](const LoopData &data, const int b, const int v, const int km,
                             const int jm, const int im) {
    return ComputeFluxCell(data.in(b, v, km, jm, im), data.fx_up(b, v, km, jm, im),
                           data.fx_lo(b, v, km, jm, im), data.fy_up(b, v, km, jm, im),
                           data.fy_lo(b, v, km, jm, im), data.fz_up(b, v, km, jm, im),
                           data.fz_lo(b, v, km, jm, im));
  });
}

void RunCpuSIMDHeavy(const Dataset &dataset, const int heavy_iterations) {
  RunCpuSIMDLoop(dataset, [heavy_iterations](const LoopData &data, const int b, const int v,
                                             const int km, const int jm, const int im) {
    return ComputeHeavyCell(data.in(b, v, km, jm, im), data.aux(b, v, km, jm, im),
                            heavy_iterations);
  });
}

void RunCpuSIMDStencil(const Dataset &dataset) {
  const auto &shape = dataset.problem;
  const auto &data = dataset.data;
  for (int b = 0; b < shape.blocks; ++b) {
    const int nvar = data.active_counts(b);
    for (int v = 0; v < nvar; ++v) {
      for (int km = shape.domain_k.s; km <= shape.domain_k.e; ++km) {
        const int km_m1 = shape.ndim > 2 ? km - 1 : km;
        const int km_p1 = shape.ndim > 2 ? km + 1 : km;
        for (int jm = shape.domain_j.s; jm <= shape.domain_j.e; ++jm) {
          const int jm_m1 = shape.ndim > 1 ? jm - 1 : jm;
          const int jm_p1 = shape.ndim > 1 ? jm + 1 : jm;
#pragma omp simd
          for (int im = shape.domain_i.s; im <= shape.domain_i.e; ++im) {
            data.out(b, v, km, jm, im) =
                ComputeStencilCell(data.in(b, v, km, jm, im), data.in(b, v, km, jm, im - 1),
                                   data.in(b, v, km, jm, im + 1),
                                   data.in(b, v, km, jm_m1, im),
                                   data.in(b, v, km, jm_p1, im),
                                   data.in(b, v, km_m1, jm, im),
                                   data.in(b, v, km_p1, jm, im), data.aux(b, v, km, jm, im),
                                   data.fx_up(b, v, km, jm, im), data.fx_lo(b, v, km, jm, im));
          }
        }
      }
    }
  }
}

template <typename Body>
void RunCpuRowVarSIMDLoop(const Dataset &dataset, Body body) {
  const auto &shape = dataset.problem;
  const auto &data = dataset.data;
  for (int b = 0; b < shape.blocks; ++b) {
    const int nvar = data.active_counts(b);
    for (int km = shape.domain_k.s; km <= shape.domain_k.e; ++km) {
      for (int jm = shape.domain_j.s; jm <= shape.domain_j.e; ++jm) {
        for (int v = 0; v < nvar; ++v) {
#pragma omp simd
          for (int im = shape.domain_i.s; im <= shape.domain_i.e; ++im) {
            data.out(b, v, km, jm, im) = body(data, b, v, km, jm, im);
          }
        }
      }
    }
  }
}

void RunCpuRowVarSIMDLight(const Dataset &dataset) {
  RunCpuRowVarSIMDLoop(dataset, [](const LoopData &data, const int b, const int v, const int km,
                                   const int jm, const int im) {
    return ComputeLightCell(data.in(b, v, km, jm, im), data.aux(b, v, km, jm, im));
  });
}

void RunCpuRowVarSIMDFlux(const Dataset &dataset) {
  RunCpuRowVarSIMDLoop(dataset, [](const LoopData &data, const int b, const int v, const int km,
                                   const int jm, const int im) {
    return ComputeFluxCell(data.in(b, v, km, jm, im), data.fx_up(b, v, km, jm, im),
                           data.fx_lo(b, v, km, jm, im), data.fy_up(b, v, km, jm, im),
                           data.fy_lo(b, v, km, jm, im), data.fz_up(b, v, km, jm, im),
                           data.fz_lo(b, v, km, jm, im));
  });
}

void RunCpuRowVarSIMDHeavy(const Dataset &dataset, const int heavy_iterations) {
  RunCpuRowVarSIMDLoop(dataset,
                       [heavy_iterations](const LoopData &data, const int b, const int v,
                                          const int km, const int jm, const int im) {
                         return ComputeHeavyCell(data.in(b, v, km, jm, im),
                                                 data.aux(b, v, km, jm, im), heavy_iterations);
                       });
}

void RunCpuRowVarSIMDStencil(const Dataset &dataset) {
  const auto &shape = dataset.problem;
  const auto &data = dataset.data;
  for (int b = 0; b < shape.blocks; ++b) {
    const int nvar = data.active_counts(b);
    for (int km = shape.domain_k.s; km <= shape.domain_k.e; ++km) {
      const int km_m1 = shape.ndim > 2 ? km - 1 : km;
      const int km_p1 = shape.ndim > 2 ? km + 1 : km;
      for (int jm = shape.domain_j.s; jm <= shape.domain_j.e; ++jm) {
        const int jm_m1 = shape.ndim > 1 ? jm - 1 : jm;
        const int jm_p1 = shape.ndim > 1 ? jm + 1 : jm;
        for (int v = 0; v < nvar; ++v) {
#pragma omp simd
          for (int im = shape.domain_i.s; im <= shape.domain_i.e; ++im) {
            data.out(b, v, km, jm, im) =
                ComputeStencilCell(data.in(b, v, km, jm, im), data.in(b, v, km, jm, im - 1),
                                   data.in(b, v, km, jm, im + 1),
                                   data.in(b, v, km, jm_m1, im),
                                   data.in(b, v, km, jm_p1, im),
                                   data.in(b, v, km_m1, jm, im),
                                   data.in(b, v, km_p1, jm, im), data.aux(b, v, km, jm, im),
                                   data.fx_up(b, v, km, jm, im), data.fx_lo(b, v, km, jm, im));
          }
        }
      }
    }
  }
}

template <typename Body>
void RunCpuHierarchicalLoop(const Dataset &dataset, const RawMemoryIndexer &idxer, Body body) {
  const auto &shape = dataset.problem;
  auto data = dataset.data;
  for (int b = 0; b < shape.blocks; ++b) {
    const int nvar = data.active_counts(b);
    for (int idx_out = 0; idx_out < idxer.GetNouter(); ++idx_out) {
      const auto [ks, js, is] = idxer.GetStartIndices(idx_out);
      const int ninner = idxer.GetNinnerRaw(idx_out);
      for (int v = 0; v < nvar; ++v) {
        const double *const in_ptr = &data.in(b, v, ks, js, is);
        const double *const aux_ptr = &data.aux(b, v, ks, js, is);
        const double *const fx_up_ptr = &data.fx_up(b, v, ks, js, is);
        const double *const fx_lo_ptr = &data.fx_lo(b, v, ks, js, is);
        const double *const fy_up_ptr = &data.fy_up(b, v, ks, js, is);
        const double *const fy_lo_ptr = &data.fy_lo(b, v, ks, js, is);
        const double *const fz_up_ptr = &data.fz_up(b, v, ks, js, is);
        const double *const fz_lo_ptr = &data.fz_lo(b, v, ks, js, is);
        double *const out_ptr = &data.out(b, v, ks, js, is);
#pragma omp simd
        for (int idx = 0; idx < ninner; ++idx) {
          out_ptr[idx] = body(in_ptr, aux_ptr, fx_up_ptr, fx_lo_ptr, fy_up_ptr, fy_lo_ptr,
                              fz_up_ptr, fz_lo_ptr, idx);
        }
      }
    }
  }
}

void RunCpuHierarchicalLight(const Dataset &dataset, const RawMemoryIndexer &idxer) {
  RunCpuHierarchicalLoop(dataset, idxer,
                         [](const double *in_ptr, const double *aux_ptr, const double *,
                            const double *, const double *, const double *, const double *,
                            const double *, const int idx) {
                           return ComputeLightCell(in_ptr[idx], aux_ptr[idx]);
                         });
}

void RunCpuHierarchicalFlux(const Dataset &dataset, const RawMemoryIndexer &idxer) {
  RunCpuHierarchicalLoop(dataset, idxer,
                         [](const double *in_ptr, const double *, const double *fx_up_ptr,
                            const double *fx_lo_ptr, const double *fy_up_ptr,
                            const double *fy_lo_ptr, const double *fz_up_ptr,
                            const double *fz_lo_ptr, const int idx) {
                           return ComputeFluxCell(in_ptr[idx], fx_up_ptr[idx], fx_lo_ptr[idx],
                                                  fy_up_ptr[idx], fy_lo_ptr[idx],
                                                  fz_up_ptr[idx], fz_lo_ptr[idx]);
                         });
}

void RunCpuHierarchicalHeavy(const Dataset &dataset, const RawMemoryIndexer &idxer,
                             const int heavy_iterations) {
  RunCpuHierarchicalLoop(dataset, idxer,
                         [heavy_iterations](const double *in_ptr, const double *aux_ptr,
                                            const double *, const double *, const double *,
                                            const double *, const double *, const double *,
                                            const int idx) {
                           return ComputeHeavyCell(in_ptr[idx], aux_ptr[idx], heavy_iterations);
                         });
}

void RunCpuHierarchicalStencil(const Dataset &dataset, const RawMemoryIndexer &idxer) {
  const auto &shape = dataset.problem;
  auto data = dataset.data;
  const int j_stride = shape.ndim > 1 ? MemoryJStride(shape) : 0;
  const int k_stride = shape.ndim > 2 ? MemoryKStride(shape) : 0;
  for (int b = 0; b < shape.blocks; ++b) {
    const int nvar = data.active_counts(b);
    for (int idx_out = 0; idx_out < idxer.GetNouter(); ++idx_out) {
      const auto [ks, js, is] = idxer.GetStartIndices(idx_out);
      const int ninner = idxer.GetNinnerRaw(idx_out);
      for (int v = 0; v < nvar; ++v) {
        const double *const in_ptr = &data.in(b, v, ks, js, is);
        const double *const aux_ptr = &data.aux(b, v, ks, js, is);
        const double *const fx_up_ptr = &data.fx_up(b, v, ks, js, is);
        const double *const fx_lo_ptr = &data.fx_lo(b, v, ks, js, is);
        double *const out_ptr = &data.out(b, v, ks, js, is);
#pragma omp simd
        for (int idx = 0; idx < ninner; ++idx) {
          out_ptr[idx] =
              ComputeStencilCell(in_ptr[idx], in_ptr[idx - 1], in_ptr[idx + 1],
                                 in_ptr[idx - j_stride], in_ptr[idx + j_stride],
                                 in_ptr[idx - k_stride], in_ptr[idx + k_stride], aux_ptr[idx],
                                 fx_up_ptr[idx], fx_lo_ptr[idx]);
        }
      }
    }
  }
}

template <typename Body>
void RunCpuCoalescedOuterVarLoop(const Dataset &dataset, const RawMemoryIndexer &idxer, Body body) {
  const auto &shape = dataset.problem;
  auto data = dataset.data;
  for (int b = 0; b < shape.blocks; ++b) {
    const int nvar = data.active_counts(b);
    for (int v = 0; v < nvar; ++v) {
      for (int idx_out = 0; idx_out < idxer.GetNouter(); ++idx_out) {
        const auto [ks, js, is] = idxer.GetStartIndices(idx_out);
        const int ninner = idxer.GetNinnerRaw(idx_out);
        const double *const in_ptr = &data.in(b, v, ks, js, is);
        const double *const aux_ptr = &data.aux(b, v, ks, js, is);
        const double *const fx_up_ptr = &data.fx_up(b, v, ks, js, is);
        const double *const fx_lo_ptr = &data.fx_lo(b, v, ks, js, is);
        const double *const fy_up_ptr = &data.fy_up(b, v, ks, js, is);
        const double *const fy_lo_ptr = &data.fy_lo(b, v, ks, js, is);
        const double *const fz_up_ptr = &data.fz_up(b, v, ks, js, is);
        const double *const fz_lo_ptr = &data.fz_lo(b, v, ks, js, is);
        double *const out_ptr = &data.out(b, v, ks, js, is);
#pragma omp simd
        for (int idx = 0; idx < ninner; ++idx) {
          out_ptr[idx] = body(in_ptr, aux_ptr, fx_up_ptr, fx_lo_ptr, fy_up_ptr, fy_lo_ptr,
                              fz_up_ptr, fz_lo_ptr, idx);
        }
      }
    }
  }
}

void RunCpuCoalescedOuterVarLight(const Dataset &dataset, const RawMemoryIndexer &idxer) {
  RunCpuCoalescedOuterVarLoop(dataset, idxer,
                              [](const double *in_ptr, const double *aux_ptr, const double *,
                                 const double *, const double *, const double *, const double *,
                                 const double *, const int idx) {
                                return ComputeLightCell(in_ptr[idx], aux_ptr[idx]);
                              });
}

void RunCpuCoalescedOuterVarFlux(const Dataset &dataset, const RawMemoryIndexer &idxer) {
  RunCpuCoalescedOuterVarLoop(dataset, idxer,
                              [](const double *in_ptr, const double *, const double *fx_up_ptr,
                                 const double *fx_lo_ptr, const double *fy_up_ptr,
                                 const double *fy_lo_ptr, const double *fz_up_ptr,
                                 const double *fz_lo_ptr, const int idx) {
                                return ComputeFluxCell(in_ptr[idx], fx_up_ptr[idx],
                                                       fx_lo_ptr[idx], fy_up_ptr[idx],
                                                       fy_lo_ptr[idx], fz_up_ptr[idx],
                                                       fz_lo_ptr[idx]);
                              });
}

void RunCpuCoalescedOuterVarHeavy(const Dataset &dataset, const RawMemoryIndexer &idxer,
                                  const int heavy_iterations) {
  RunCpuCoalescedOuterVarLoop(dataset, idxer,
                              [heavy_iterations](const double *in_ptr, const double *aux_ptr,
                                                 const double *, const double *, const double *,
                                                 const double *, const double *, const double *,
                                                 const int idx) {
                                return ComputeHeavyCell(in_ptr[idx], aux_ptr[idx],
                                                        heavy_iterations);
                              });
}

void RunCpuCoalescedOuterVarStencil(const Dataset &dataset, const RawMemoryIndexer &idxer) {
  const auto &shape = dataset.problem;
  auto data = dataset.data;
  const int j_stride = shape.ndim > 1 ? MemoryJStride(shape) : 0;
  const int k_stride = shape.ndim > 2 ? MemoryKStride(shape) : 0;
  for (int b = 0; b < shape.blocks; ++b) {
    const int nvar = data.active_counts(b);
    for (int v = 0; v < nvar; ++v) {
      for (int idx_out = 0; idx_out < idxer.GetNouter(); ++idx_out) {
        const auto [ks, js, is] = idxer.GetStartIndices(idx_out);
        const int ninner = idxer.GetNinnerRaw(idx_out);
        const double *const in_ptr = &data.in(b, v, ks, js, is);
        const double *const aux_ptr = &data.aux(b, v, ks, js, is);
        const double *const fx_up_ptr = &data.fx_up(b, v, ks, js, is);
        const double *const fx_lo_ptr = &data.fx_lo(b, v, ks, js, is);
        double *const out_ptr = &data.out(b, v, ks, js, is);
#pragma omp simd
        for (int idx = 0; idx < ninner; ++idx) {
          out_ptr[idx] =
              ComputeStencilCell(in_ptr[idx], in_ptr[idx - 1], in_ptr[idx + 1],
                                 in_ptr[idx - j_stride], in_ptr[idx + j_stride],
                                 in_ptr[idx - k_stride], in_ptr[idx + k_stride], aux_ptr[idx],
                                 fx_up_ptr[idx], fx_lo_ptr[idx]);
        }
      }
    }
  }
}

void RunFlatRangeLight(const Dataset &dataset) {
  const auto shape = dataset.problem;
  const auto data = dataset.data;
  Kokkos::parallel_for(
      "FlatRangeLight", Kokkos::RangePolicy<>(0, static_cast<int>(shape.cell_indexer.size())),
      KOKKOS_LAMBDA(const int outer_idx) {
        const auto [b, km, jm, im] = shape.cell_indexer(outer_idx);
        for (int v = 0; v < data.active_counts(b); ++v) {
          data.out(b, v, km, jm, im) =
              ComputeLightCell(data.in(b, v, km, jm, im), data.aux(b, v, km, jm, im));
        }
      });
}

void RunFlatRangeFlux(const Dataset &dataset) {
  const auto shape = dataset.problem;
  const auto data = dataset.data;
  Kokkos::parallel_for(
      "FlatRangeFlux", Kokkos::RangePolicy<>(0, static_cast<int>(shape.cell_indexer.size())),
      KOKKOS_LAMBDA(const int outer_idx) {
        const auto [b, km, jm, im] = shape.cell_indexer(outer_idx);
        for (int v = 0; v < data.active_counts(b); ++v) {
          data.out(b, v, km, jm, im) =
              ComputeFluxCell(data.in(b, v, km, jm, im), data.fx_up(b, v, km, jm, im),
                              data.fx_lo(b, v, km, jm, im), data.fy_up(b, v, km, jm, im),
                              data.fy_lo(b, v, km, jm, im), data.fz_up(b, v, km, jm, im),
                              data.fz_lo(b, v, km, jm, im));
        }
      });
}

void RunFlatRangeHeavy(const Dataset &dataset, int heavy_iterations) {
  const auto shape = dataset.problem;
  const auto data = dataset.data;
  Kokkos::parallel_for(
      "FlatRangeHeavy", Kokkos::RangePolicy<>(0, static_cast<int>(shape.cell_indexer.size())),
      KOKKOS_LAMBDA(const int outer_idx) {
        const auto [b, km, jm, im] = shape.cell_indexer(outer_idx);
        for (int v = 0; v < data.active_counts(b); ++v) {
          data.out(b, v, km, jm, im) =
              ComputeHeavyCell(data.in(b, v, km, jm, im), data.aux(b, v, km, jm, im),
                               heavy_iterations);
        }
      });
}

void RunFlatRangeStencil(const Dataset &dataset) {
  const auto shape = dataset.problem;
  const auto data = dataset.data;
  Kokkos::parallel_for(
      "FlatRangeStencil", Kokkos::RangePolicy<>(0, static_cast<int>(shape.cell_indexer.size())),
      KOKKOS_LAMBDA(const int outer_idx) {
        const auto [b, km, jm, im] = shape.cell_indexer(outer_idx);
        const int km_m1 = shape.ndim > 2 ? km - 1 : km;
        const int km_p1 = shape.ndim > 2 ? km + 1 : km;
        const int jm_m1 = shape.ndim > 1 ? jm - 1 : jm;
        const int jm_p1 = shape.ndim > 1 ? jm + 1 : jm;
        for (int v = 0; v < data.active_counts(b); ++v) {
          data.out(b, v, km, jm, im) =
              ComputeStencilCell(data.in(b, v, km, jm, im), data.in(b, v, km, jm, im - 1),
                                 data.in(b, v, km, jm, im + 1), data.in(b, v, km, jm_m1, im),
                                 data.in(b, v, km, jm_p1, im), data.in(b, v, km_m1, jm, im),
                                 data.in(b, v, km_p1, jm, im), data.aux(b, v, km, jm, im),
                                 data.fx_up(b, v, km, jm, im), data.fx_lo(b, v, km, jm, im));
        }
      });
}

void RunMDRangeLight(const Dataset &dataset) {
  const auto shape = dataset.problem;
  const auto data = dataset.data;
  Kokkos::parallel_for(
      "MDRangeLight",
      MDRange4({0, shape.domain_k.s, shape.domain_j.s, shape.domain_i.s},
               {shape.blocks, shape.domain_k.e + 1, shape.domain_j.e + 1,
                shape.domain_i.e + 1}),
      KOKKOS_LAMBDA(const int b, const int km, const int jm, const int im) {
        for (int v = 0; v < data.active_counts(b); ++v) {
          data.out(b, v, km, jm, im) =
              ComputeLightCell(data.in(b, v, km, jm, im), data.aux(b, v, km, jm, im));
        }
      });
}

void RunMDRangeFlux(const Dataset &dataset) {
  const auto shape = dataset.problem;
  const auto data = dataset.data;
  Kokkos::parallel_for(
      "MDRangeFlux",
      MDRange4({0, shape.domain_k.s, shape.domain_j.s, shape.domain_i.s},
               {shape.blocks, shape.domain_k.e + 1, shape.domain_j.e + 1,
                shape.domain_i.e + 1}),
      KOKKOS_LAMBDA(const int b, const int km, const int jm, const int im) {
        for (int v = 0; v < data.active_counts(b); ++v) {
          data.out(b, v, km, jm, im) =
              ComputeFluxCell(data.in(b, v, km, jm, im), data.fx_up(b, v, km, jm, im),
                              data.fx_lo(b, v, km, jm, im), data.fy_up(b, v, km, jm, im),
                              data.fy_lo(b, v, km, jm, im), data.fz_up(b, v, km, jm, im),
                              data.fz_lo(b, v, km, jm, im));
        }
      });
}

void RunMDRangeHeavy(const Dataset &dataset, int heavy_iterations) {
  const auto shape = dataset.problem;
  const auto data = dataset.data;
  Kokkos::parallel_for(
      "MDRangeHeavy",
      MDRange4({0, shape.domain_k.s, shape.domain_j.s, shape.domain_i.s},
               {shape.blocks, shape.domain_k.e + 1, shape.domain_j.e + 1,
                shape.domain_i.e + 1}),
      KOKKOS_LAMBDA(const int b, const int km, const int jm, const int im) {
        for (int v = 0; v < data.active_counts(b); ++v) {
          data.out(b, v, km, jm, im) =
              ComputeHeavyCell(data.in(b, v, km, jm, im), data.aux(b, v, km, jm, im),
                               heavy_iterations);
        }
      });
}

void RunMDRangeStencil(const Dataset &dataset) {
  const auto shape = dataset.problem;
  const auto data = dataset.data;
  Kokkos::parallel_for(
      "MDRangeStencil",
      MDRange4({0, shape.domain_k.s, shape.domain_j.s, shape.domain_i.s},
               {shape.blocks, shape.domain_k.e + 1, shape.domain_j.e + 1,
                shape.domain_i.e + 1}),
      KOKKOS_LAMBDA(const int b, const int km, const int jm, const int im) {
        const int km_m1 = shape.ndim > 2 ? km - 1 : km;
        const int km_p1 = shape.ndim > 2 ? km + 1 : km;
        const int jm_m1 = shape.ndim > 1 ? jm - 1 : jm;
        const int jm_p1 = shape.ndim > 1 ? jm + 1 : jm;
        for (int v = 0; v < data.active_counts(b); ++v) {
          data.out(b, v, km, jm, im) =
              ComputeStencilCell(data.in(b, v, km, jm, im), data.in(b, v, km, jm, im - 1),
                                 data.in(b, v, km, jm, im + 1), data.in(b, v, km, jm_m1, im),
                                 data.in(b, v, km, jm_p1, im), data.in(b, v, km_m1, jm, im),
                                 data.in(b, v, km_p1, jm, im), data.aux(b, v, km, jm, im),
                                 data.fx_up(b, v, km, jm, im), data.fx_lo(b, v, km, jm, im));
        }
      });
}

template <typename Body>
void RunHierarchicalLoop(const Dataset &dataset, const RawMemoryIndexer &idxer, int team_size,
                         const char *label, Body body) {
  const auto shape = dataset.problem;
  const auto data = dataset.data;
  const int league_size = shape.blocks * idxer.GetNouter();
  const TeamPolicy policy =
      team_size > 0 ? TeamPolicy(league_size, team_size) : TeamPolicy(league_size, Kokkos::AUTO);

  Kokkos::parallel_for(
      label, policy, KOKKOS_LAMBDA(const TeamMember &member) {
        const int league = member.league_rank();
        const int b = league / idxer.GetNouter();
        const int idx_out = league % idxer.GetNouter();
        const auto [ks, js, is] = idxer.GetStartIndices(idx_out);
        const int ninner = idxer.GetNinnerRaw(idx_out);
        for (int v = 0; v < data.active_counts(b); ++v) {
          const double *const in_ptr = &data.in(b, v, ks, js, is);
          const double *const aux_ptr = &data.aux(b, v, ks, js, is);
          const double *const fx_up_ptr = &data.fx_up(b, v, ks, js, is);
          const double *const fx_lo_ptr = &data.fx_lo(b, v, ks, js, is);
          const double *const fy_up_ptr = &data.fy_up(b, v, ks, js, is);
          const double *const fy_lo_ptr = &data.fy_lo(b, v, ks, js, is);
          const double *const fz_up_ptr = &data.fz_up(b, v, ks, js, is);
          const double *const fz_lo_ptr = &data.fz_lo(b, v, ks, js, is);
          double *const out_ptr = &data.out(b, v, ks, js, is);

          Kokkos::parallel_for(Kokkos::TeamThreadRange(member, 0, ninner), [&](const int idx) {
            out_ptr[idx] = body(in_ptr, aux_ptr, fx_up_ptr, fx_lo_ptr, fy_up_ptr, fy_lo_ptr,
                                fz_up_ptr, fz_lo_ptr, idx);
          });
          member.team_barrier();
        }
      });
}

void RunHierarchicalLight(const Dataset &dataset, const RawMemoryIndexer &idxer, int team_size) {
  RunHierarchicalLoop(
      dataset, idxer, team_size, "HierarchicalLight",
      KOKKOS_LAMBDA(const double *in_ptr, const double *aux_ptr, const double *,
                    const double *, const double *, const double *, const double *,
                    const double *, const int idx) {
        return ComputeLightCell(in_ptr[idx], aux_ptr[idx]);
      });
}

void RunHierarchicalFlux(const Dataset &dataset, const RawMemoryIndexer &idxer, int team_size) {
  RunHierarchicalLoop(
      dataset, idxer, team_size, "HierarchicalFlux",
      KOKKOS_LAMBDA(const double *in_ptr, const double *, const double *fx_up_ptr,
                    const double *fx_lo_ptr, const double *fy_up_ptr,
                    const double *fy_lo_ptr, const double *fz_up_ptr,
                    const double *fz_lo_ptr, const int idx) {
        return ComputeFluxCell(in_ptr[idx], fx_up_ptr[idx], fx_lo_ptr[idx], fy_up_ptr[idx],
                               fy_lo_ptr[idx], fz_up_ptr[idx], fz_lo_ptr[idx]);
      });
}

void RunHierarchicalHeavy(const Dataset &dataset, const RawMemoryIndexer &idxer, int team_size,
                          int heavy_iterations) {
  RunHierarchicalLoop(
      dataset, idxer, team_size, "HierarchicalHeavy",
      KOKKOS_LAMBDA(const double *in_ptr, const double *aux_ptr, const double *,
                    const double *, const double *, const double *, const double *,
                    const double *, const int idx) {
        return ComputeHeavyCell(in_ptr[idx], aux_ptr[idx], heavy_iterations);
      });
}

void RunHierarchicalStencil(const Dataset &dataset, const RawMemoryIndexer &idxer, int team_size) {
  const auto shape = dataset.problem;
  const int j_stride = shape.ndim > 1 ? MemoryJStride(shape) : 0;
  const int k_stride = shape.ndim > 2 ? MemoryKStride(shape) : 0;
  RunHierarchicalLoop(
      dataset, idxer, team_size, "HierarchicalStencil",
      KOKKOS_LAMBDA(const double *in_ptr, const double *aux_ptr, const double *fx_up_ptr,
                    const double *fx_lo_ptr, const double *, const double *, const double *,
                    const double *, const int idx) {
        return ComputeStencilCell(in_ptr[idx], in_ptr[idx - 1], in_ptr[idx + 1],
                                  in_ptr[idx - j_stride], in_ptr[idx + j_stride],
                                  in_ptr[idx - k_stride], in_ptr[idx + k_stride], aux_ptr[idx],
                                  fx_up_ptr[idx], fx_lo_ptr[idx]);
      });
}

int RequestedTeamSize(const BenchmarkConfig &config) {
  if (config.team_size_mode == "explicit" && config.explicit_team_size > 0) {
    return config.explicit_team_size;
  }
  return 0;
}

RawMemoryIndexer SelectedCpuHierarchicalIndexer(const Dataset &dataset,
                                                const BenchmarkConfig &config) {
  if (config.inner_chunk_length > 0) {
    return BuildTunedIndexer(dataset.problem, config);
  }
  return BuildRawMemoryIndexer(dataset.problem, dataset.problem.ni * dataset.problem.nj);
}

}  // namespace

Dataset BuildDataset(const BenchmarkConfig &config) {
  Dataset dataset;
  dataset.problem = BuildProblemShape(config);
  InitializeLoopData(dataset.problem, &dataset.data);
  return dataset;
}

void PrepareDataset(const BenchmarkConfig &config, const RaggedMetadata &metadata, Dataset *dataset) {
  SetActiveCounts(config, metadata, &dataset->data);
}

void ExecuteLoopPattern(const BenchmarkConfig &config, const RaggedMetadata &metadata,
                        Dataset *dataset) {
  (void)metadata;
  switch (config.variant) {
    case VariantKind::Flat:
      switch (config.kernel) {
        case KernelKind::Light:
          RunFlatRangeLight(*dataset);
          break;
        case KernelKind::Flux:
          RunFlatRangeFlux(*dataset);
          break;
        case KernelKind::Stencil:
          RunFlatRangeStencil(*dataset);
          break;
        case KernelKind::Heavy:
          RunFlatRangeHeavy(*dataset, config.heavy_iterations);
          break;
      }
      break;
    case VariantKind::MDRange:
      switch (config.kernel) {
        case KernelKind::Light:
          RunMDRangeLight(*dataset);
          break;
        case KernelKind::Flux:
          RunMDRangeFlux(*dataset);
          break;
        case KernelKind::Stencil:
          RunMDRangeStencil(*dataset);
          break;
        case KernelKind::Heavy:
          RunMDRangeHeavy(*dataset, config.heavy_iterations);
          break;
      }
      break;
    case VariantKind::Hierarchical:
      {
      const auto idxer = BuildRawMemoryIndexer(
          dataset->problem, dataset->problem.ni * dataset->problem.nj);
      switch (config.kernel) {
        case KernelKind::Light:
          RunHierarchicalLight(*dataset, idxer, RequestedTeamSize(config));
          break;
        case KernelKind::Flux:
          RunHierarchicalFlux(*dataset, idxer, RequestedTeamSize(config));
          break;
        case KernelKind::Stencil:
          RunHierarchicalStencil(*dataset, idxer, RequestedTeamSize(config));
          break;
        case KernelKind::Heavy:
          RunHierarchicalHeavy(*dataset, idxer, RequestedTeamSize(config),
                               config.heavy_iterations);
          break;
      }
      break;
      }
    case VariantKind::Tuned:
      {
      const auto idxer = BuildTunedIndexer(dataset->problem, config);
      switch (config.kernel) {
        case KernelKind::Light:
          RunHierarchicalLight(*dataset, idxer, RequestedTeamSize(config));
          break;
        case KernelKind::Flux:
          RunHierarchicalFlux(*dataset, idxer, RequestedTeamSize(config));
          break;
        case KernelKind::Stencil:
          RunHierarchicalStencil(*dataset, idxer, RequestedTeamSize(config));
          break;
        case KernelKind::Heavy:
          RunHierarchicalHeavy(*dataset, idxer, RequestedTeamSize(config),
                               config.heavy_iterations);
          break;
      }
      break;
      }
    case VariantKind::CpuSIMD:
      switch (config.kernel) {
        case KernelKind::Light:
          RunCpuSIMDLight(*dataset);
          break;
        case KernelKind::Flux:
          RunCpuSIMDFlux(*dataset);
          break;
        case KernelKind::Stencil:
          RunCpuSIMDStencil(*dataset);
          break;
        case KernelKind::Heavy:
          RunCpuSIMDHeavy(*dataset, config.heavy_iterations);
          break;
      }
      break;
    case VariantKind::CpuHierarchical: {
      const auto &idxer = SelectedCpuHierarchicalIndexer(*dataset, config);
      switch (config.kernel) {
        case KernelKind::Light:
          RunCpuHierarchicalLight(*dataset, idxer);
          break;
        case KernelKind::Flux:
          RunCpuHierarchicalFlux(*dataset, idxer);
          break;
        case KernelKind::Stencil:
          RunCpuHierarchicalStencil(*dataset, idxer);
          break;
        case KernelKind::Heavy:
          RunCpuHierarchicalHeavy(*dataset, idxer, config.heavy_iterations);
          break;
      }
      break;
    }
    case VariantKind::CpuCoalescedOuterVar: {
      const auto &idxer = SelectedCpuHierarchicalIndexer(*dataset, config);
      switch (config.kernel) {
        case KernelKind::Light:
          RunCpuCoalescedOuterVarLight(*dataset, idxer);
          break;
        case KernelKind::Flux:
          RunCpuCoalescedOuterVarFlux(*dataset, idxer);
          break;
        case KernelKind::Stencil:
          RunCpuCoalescedOuterVarStencil(*dataset, idxer);
          break;
        case KernelKind::Heavy:
          RunCpuCoalescedOuterVarHeavy(*dataset, idxer, config.heavy_iterations);
          break;
      }
      break;
    }
    case VariantKind::CpuRowVarSIMD:
      switch (config.kernel) {
        case KernelKind::Light:
          RunCpuRowVarSIMDLight(*dataset);
          break;
        case KernelKind::Flux:
          RunCpuRowVarSIMDFlux(*dataset);
          break;
        case KernelKind::Stencil:
          RunCpuRowVarSIMDStencil(*dataset);
          break;
        case KernelKind::Heavy:
          RunCpuRowVarSIMDHeavy(*dataset, config.heavy_iterations);
          break;
      }
      break;
  }
  Kokkos::fence();
}

std::uint64_t CountUpdates(const BenchmarkConfig &config, const RaggedMetadata &metadata) {
  std::uint64_t total = 0;
  const std::uint64_t cells =
      static_cast<std::uint64_t>(config.nk) * static_cast<std::uint64_t>(config.nj) *
      static_cast<std::uint64_t>(config.ni);
  for (int block = 0; block < config.blocks; ++block) {
    total += static_cast<std::uint64_t>(
                 ActiveVariablesForBlock(metadata, config.ragged, block, config.variables)) *
             cells;
  }
  return total;
}

int EffectiveInnerChunkLength(const BenchmarkConfig &config) {
  if (config.variant == VariantKind::Flat || config.variant == VariantKind::MDRange ||
      config.variant == VariantKind::CpuSIMD || config.variant == VariantKind::CpuRowVarSIMD) {
    return 0;
  }
  if (config.variant == VariantKind::Tuned) {
    return static_cast<int>(DefaultTunedChunkLength(config.ni, config.inner_chunk_length));
  }
  if ((config.variant == VariantKind::CpuHierarchical ||
       config.variant == VariantKind::CpuCoalescedOuterVar) &&
      config.inner_chunk_length > 0) {
    return static_cast<int>(DefaultTunedChunkLength(config.ni, config.inner_chunk_length));
  }
  return static_cast<int>(DefaultHierarchicalChunkLength(config.ni, config.nj,
                                                         config.inner_chunk_length));
}

double EstimatedBytesPerUpdate(KernelKind kind) {
  if (kind == KernelKind::Light || kind == KernelKind::Heavy) {
    return 3.0 * sizeof(double);
  }
  if (kind == KernelKind::Stencil) {
    return 10.0 * sizeof(double);
  }
  return 8.0 * sizeof(double);
}

}  // namespace plb
