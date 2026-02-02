#define CATCH_CONFIG_RUNNER
#include <catch2/catch_all.hpp>

#include <Kokkos_Core.hpp>

int main(int argc, char *argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);
  Catch::Session session;
  // You can customize the session here if needed, e.g.:
  // session.configData().showSuccessfulTests = true;
  return session.run(argc, argv);
}