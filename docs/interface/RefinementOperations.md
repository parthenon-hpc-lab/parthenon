# Prolongation and Restriction Operations

When mesh refinement is enabled, variables between different
refinement regions must be communicated via prolongation and
restriction. The default operations used for cell-centered variables
are averaging weighted by cell volume for restriction and linear
interpolation with minmod limiting for prolongation.

## User-Defined Operations

A user may define their own prolongation and restriction
operations. To do so, you must define a struct, templated on
dimension, containing only a void function `Do` with the following signature:

```C++
KOKKOS_FORCEINLINE_FUNCTION static void
Do(const int l, const int m, const int n, const int ck, const int cj, const int ci,
   const IndexRange &ckb, const IndexRange &cjb, const IndexRange &cib,
   const IndexRange &kb, const IndexRange &jb, const IndexRange &ib,
   const Coordinates_t &coords, const Coordinates_t &coarse_coords,
   const ParArray6D<Real> *pcoarse, const ParArray6D<Real> *pfine)
```

where `l`, `m`, `n` are the indices of a variable object not tied to
mesh (for example the tensor indices of a rank 3 tensor). `ck`, `cj`,
and `ci` are `k`, `j`, and `i` indices of the cell on the coarse
buffer. `ckb`, `cjb`, `cib` are the `k`, `j`, and `i` indexrange
bounds for the coarse buffer. `kb`, `jb`, and `ib` are the same but on
teh fine buffer. `coords` and `coarse_coords` are the coordinates
objects on the coarse and fine buffers, and `pcoarse` and `pfine` are
pointers to the coarse and fine data for a variable on a given
meshblock.

So for example, this is part of the implmentation for the default
restrict operation:

```C++
template <int DIM>
struct RestrictCellAverage {
  KOKKOS_FORCEINLINE_FUNCTION static void
  Do(const int l, const int m, const int n, const int ck, const int cj, const int ci,
     const IndexRange &ckb, const IndexRange &cjb, const IndexRange &cib,
     const IndexRange &kb, const IndexRange &jb, const IndexRange &ib,
     const Coordinates_t &coords, const Coordinates_t &coarse_coords,
     const ParArray6D<Real> *pcoarse, const ParArray6D<Real> *pfine) {
    auto &coarse = *pcoarse;
    auto &fine = *pfine;
    const int i = (ci - cib.s) * 2 + ib.s;
    int j = jb.s;
    if constexpr (DIM > 1) {
      j = (cj - cjb.s) * 2 + jb.s;
    }
    int k = kb.s;
    if constexpr (DIM > 2) {
      k = (ck - ckb.s) * 2 + kb.s;
    }
    Real vol[2][2][2], terms[2][2][2];
    std::memset(&vol[0][0][0], 0., 8 * sizeof(Real));
    std::memset(&terms[0][0][0], 0., 8 * sizeof(Real));
    for (int ok = 0; ok < 1 + (DIM > 2); ++ok) {
      for (int oj = 0; oj < 1 + (DIM > 1); ++oj) {
        for (int oi = 0; oi < 1 + 1; ++oi) {
          vol[ok][oj][oi] = coords.Volume(k + ok, j + oj, i + oi);
          terms[ok][oj][oi] = vol[ok][oj][oi] * fine(l, m, n, k + ok, j + oj, i + oi);
        }
      }
    }
    const Real tvol = ((vol[0][0][0] + vol[0][1][0]) + (vol[0][0][1] + vol[0][1][1])) +
                      ((vol[1][0][0] + vol[1][1][0]) + (vol[1][0][1] + vol[1][1][1]));
    coarse(l, m, n, ck, cj, ci) =
        (((terms[0][0][0] + terms[0][1][0]) + (terms[0][0][1] + terms[0][1][1])) +
         ((terms[1][0][0] + terms[1][1][0]) + (terms[1][0][1] + terms[1][1][1]))) /
        tvol;
  }
};
```

This interface is the same for both prolongation and restriction,
although the implementation obviously differs.

## The default operations

The default operations for cell-centered variables are named
- `parthenon::refinement_ops::RestrictCellAverage`
- `parthenon::refinement_ops::ProlongateCellMinMod`

both structs are templated on dimension via `template<int DIM>`.

## Registering a Custom Operation

A user-defined operation must be registered for a given variable
through the variable metadata. The types are passed into the
registration function via template parameters, not through function
parameters. For example:

```C++
Metadata m({some set of flags});
m.RegisterRefinementOps<MyProlongationOp, MyRestrictionOp>();
```

You must register both prolongation and restriction together. You may,
however, use the default Parthenon structs if desired. Then any
variable registered with this metadata object will use your custom
prolongation and restriction operations.
