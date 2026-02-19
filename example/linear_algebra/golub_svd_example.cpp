#include <cassert>
#include <iostream>
#include <tuple>
#include <vector>

#include "parthenon/parthenon.hpp"

#include "linear_algebra/matrix.hpp"
#include "linear_algebra/square_svd.hpp"

int main(int argc, char *argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);
  const int N{5};
  Matrix A = Matrix::RandomGaussian(N);
  Matrix U = Matrix::Identity(N);
  Matrix V = Matrix::Identity(N);

  Matrix Ainit = A.GetDeepCopy();

  std::vector<double> sings(N, 0.0);
  SquareSVD::execute(&A, &U, &V, sings.data());

  Matrix Sigma = Matrix::Identity(N);
  for (int i = 0; i < N; ++i) {
    Sigma(i, i) = sings[i];
  }

  Matrix VT = Matrix::Transpose(V);

  Matrix temp(N), Bcheck(N);
  Multiply(U, Sigma, temp);
  Multiply(temp, VT, Bcheck);

  std::cout << Bcheck << std::endl << std::endl;

  std::cout << Ainit << std::endl;
  return 0;
}
