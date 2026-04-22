#include <iostream>
#include <string>

#include <Kokkos_Core.hpp>

#include "benchmark_driver.hpp"

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  plb::BenchmarkConfig config;
  std::string error;
  if (!plb::ParseArgs(argc, argv, &config, &error)) {
    std::cerr << error << '\n';
    if (error != plb::Usage()) {
      std::cerr << '\n' << plb::Usage();
    }
    Kokkos::finalize();
    return error == plb::Usage() ? 0 : 1;
  }
  const int result = plb::RunBenchmark(config);
  Kokkos::finalize();
  return result;
}
