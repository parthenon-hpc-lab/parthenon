#define CATCH_CONFIG_RUNNER
#include <Kokkos_Core.hpp>
#include <catch2/catch.hpp>

int main(int argc, char *argv[]) {
  // global setup...
  int result;
  Kokkos::initialize(argc, argv);
  {

    result = Catch::Session().run(argc, argv);

    // global clean-up...
  }
  Kokkos::finalize();
  return result;
}