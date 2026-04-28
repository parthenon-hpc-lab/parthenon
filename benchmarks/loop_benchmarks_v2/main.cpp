#include <iostream>

#include <Kokkos_Core.hpp>

#include "benchmark_driver.hpp"

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);

  plb2::CaseSpec spec;
  std::string error;
  if (!plb2::ParseArgs(argc, argv, &spec, &error)) {
    std::cerr << error << '\n';
    if (error == plb2::Usage()) {
      Kokkos::finalize();
      return 0;
    }
    std::cerr << '\n' << plb2::Usage();
    Kokkos::finalize();
    return 1;
  }

  const int rc = plb2::RunBenchmark(spec);
  Kokkos::finalize();
  return rc;
}
