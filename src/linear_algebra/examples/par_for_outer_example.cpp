#include <cassert>
#include <iostream>
#include <tuple>
#include <vector>

#include "parthenon/parthenon.hpp"

#include "symmetric_evd.hpp"

int main(int argc, char *argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);
  const int nmatrices{3};
  const int N{20};
  parthenon::ParArray3D<double> matrices("matrices", nmatrices, N, N);
  parthenon::ParArray2D<double> eigenvalues("eigenvalues", nmatrices, N);
  parthenon::ParArray3D<double> eigenvectors("eigenvectors", nmatrices, N, N);

  // Build matrices with known spectrum on host
  auto matrices_h = matrices.GetHostMirror();
  for (int m = 0; m < nmatrices; ++m) {
    Vector eigs(N);
    std::iota(eigs.begin(), eigs.end(), -5);
    auto MM = Matrix::FromSpectrum(eigs, 12345 + m);
    auto slice = matrices_h.Get(m);
    Kokkos::deep_copy(slice, MM.GetData());
  }

  // Copy matrices to device
  matrices.DeepCopy(matrices_h);

  const int scratch_level = 1;
  std::size_t scratch_size_in_bytes =
      2 * parthenon::ScratchPad2D<double>::shmem_size(N, N) +
      parthenon::ScratchPad1D<double>::shmem_size(N) +
      SymmetricEVD::total_shmem_scratch_size(N);
  parthenon::par_for_outer(
      DEFAULT_OUTER_LOOP_PATTERN, "QR", parthenon::DevExecSpace(),
      scratch_size_in_bytes, scratch_level, 0, nmatrices - 1,
      KOKKOS_LAMBDA(parthenon::team_mbr_t tm, const int m) {
        auto &ts = tm.team_scratch(scratch_level);
        auto mat = parthenon::ScratchPad2D<double>(ts, N, N);
        auto Q = parthenon::ScratchPad2D<double>(ts, N, N);
        auto eigs = parthenon::ScratchPad1D<double>(ts, N);

        auto lscratch = parthenon::ScratchPad1D<double>(
            ts, SymmetricEVD::double_scratch_size(N));
        auto liscratch = parthenon::ScratchPad1D<std::size_t>(
            ts, SymmetricEVD::sizet_scratch_size(N));

        // First, put the given matrix into scratch memory [In a real Gram-SVD,
        // the Gram matrices should already be in scratch after reductions over
        // tensor cores]
        parthenon::par_for_inner(
            tm, 0, N - 1, 0, N - 1,
            [&](const int r, const int c) { mat(r, c) = matrices(m, r, c); });

        // Actually calculate eigenvalues
        tm.team_barrier();
        SymmetricEVD::execute(tm, &mat, &Q, eigs.data(), lscratch.data(),
                              liscratch.data());
        tm.team_barrier();

        // Copy eigenvalues out of scratch so they can be investigated on host
        parthenon::par_for_inner(
            tm, 0, N - 1, [&](const int r) { eigenvalues(m, r) = eigs(r); });
        parthenon::par_for_inner(
            tm, 0, N - 1, 0, N - 1,
            [&](const int r, const int c) { eigenvectors(m, r, c) = Q(r, c); });
      });

  // Bring diagonal back to host and check eigenvalues
  auto eigenvalues_h = eigenvalues.GetHostMirror();
  eigenvalues_h.DeepCopy(eigenvalues);

  for (int m = 0; m < nmatrices; ++m) {
    auto d = eigenvalues_h.Get(m);
    std::vector<double> dvec(N);
    for (int i = 0; i < N; ++i)
      dvec[i] = d[i];
    std::sort(dvec.begin(), dvec.end());

    for (int i = 0; i < N; ++i)
      printf("%e ", dvec[i]);
    printf("\n");
  }

  auto eigenvectors_h = eigenvectors.GetHostMirror();
  eigenvectors_h.DeepCopy(eigenvectors);

  for (int m = 0; m < nmatrices; ++m) {
    printf("\n Matrix %i:\n", m);

    for (int e = 0; e < N; ++e) {
      std::vector<double> Av(N, 0.0);
      for (int r = 0; r < N; ++r) {
        for (int c = 0; c < N; ++c)
          Av[r] += matrices_h(m, r, c) * eigenvectors_h(m, c, e);
      }

      double mag{0.0};
      for (int r = 0; r < N; ++r) {
        Av[r] -= eigenvectors_h(m, r, e) * eigenvalues_h(m, e);
        mag += Av[r] * Av[r];
      }
      printf("||A v_{%i} - lambda_{%i} v_%i||_2 = %e\n", e, e, e,
             std::sqrt(mag));
    }
  }

  return 0;
}
