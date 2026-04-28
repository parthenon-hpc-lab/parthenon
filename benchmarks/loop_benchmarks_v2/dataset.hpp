#pragma once

#include "problem_spec.hpp"

namespace plb2 {

using View5D = Kokkos::View<double *****, Kokkos::LayoutRight>;

struct LoopData {
  View5D in;
  View5D aux;
  View5D out;
  Kokkos::View<int *> active_counts;
};

struct Dataset {
  ProblemSpec problem;
  LoopData data;
};

Dataset BuildDataset(const CaseSpec &spec);
void PrepareDataset(const CaseSpec &spec, Dataset *dataset);
std::uint64_t CountUpdates(const CaseSpec &spec, const Dataset &dataset);

}  // namespace plb2
