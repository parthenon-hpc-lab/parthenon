#include <cassert>
#include <iostream>
#include <tuple>
#include <vector>

#include "parthenon/parthenon.hpp"

#include "symmetric_evd.hpp"

int main(int argc, char *argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);
  const int nmatrices{100};
  const int N{20};
  parthenon::ParArray3D<double> matrices("matrices", nmatrices, N, N);
  parthenon::ParArray2D<double> eigenvalues("diagonal", nmatrices, N);
  parthenon::ParArray2D<double> scratch("scratch", nmatrices,
                                        SymmetricEVD::double_scratch_size(N));
  parthenon::ParArray2D<std::size_t> iscratch(
      "iscratch", nmatrices, SymmetricEVD::sizet_scratch_size(N));

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

  // Find eigenvalues of matrices on device
  parthenon::par_for(
      "QR", 0, nmatrices - 1, KOKKOS_LAMBDA(int m) {
        // First, pull out views related to matrix m
        auto cmat = matrices.Get(m);
        auto eigenvalues_local = eigenvalues.Get(m);
        auto local_scratch = scratch.Get(m);
        auto local_iscratch = iscratch.Get(m);
        // Perform the eigenvalue decomposition
        SymmetricEVD::execute(&cmat, eigenvalues_local.data(),
                              local_scratch.data(), local_iscratch.data());
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

  return 0;
}
