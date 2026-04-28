#include <iostream>
#include <string>

#include <Kokkos_Core.hpp>

#include "benchmark_driver.hpp"
#include "multicase.hpp"

namespace {

const char *FindArgValue(int argc, char **argv, const char *name) {
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == name) {
      if (i + 1 < argc) {
        return argv[i + 1];
      }
      return nullptr;
    }
  }
  return nullptr;
}

}  // namespace

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);

  const char *cases_csv = FindArgValue(argc, argv, "--cases");
  if (cases_csv != nullptr) {
    const char *results_csv = FindArgValue(argc, argv, "--csv-out");
    std::string error;
    const std::string output = results_csv != nullptr ? results_csv : "results.csv";
    const bool ok = plb2::RunCaseMatrix(cases_csv, output, &error);
    if (!ok) {
      std::cerr << error << '\n';
      Kokkos::finalize();
      return 1;
    }
    Kokkos::finalize();
    return 0;
  }

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
