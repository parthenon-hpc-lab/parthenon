#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

namespace plb {

struct RaggedMetadata {
  std::vector<int> active_counts;
};

inline RaggedMetadata BuildRaggedMetadata(int blocks, int variables, int active_min, int active_max) {
  RaggedMetadata metadata;
  metadata.active_counts.resize(blocks, variables);
  const int clamped_min = std::max(1, std::min(active_min, variables));
  const int clamped_max = std::max(clamped_min, std::min(active_max, variables));
  for (int block = 0; block < blocks; ++block) {
    const std::uint32_t hash = static_cast<std::uint32_t>(block * 1103515245u + 12345u);
    const int span = clamped_max - clamped_min + 1;
    metadata.active_counts[block] = clamped_min + static_cast<int>(hash % static_cast<std::uint32_t>(span));
  }
  return metadata;
}

inline int ActiveVariablesForBlock(const RaggedMetadata &metadata, bool ragged, int block, int fallback) {
  if (!ragged) {
    return fallback;
  }
  return metadata.active_counts[block];
}

}  // namespace plb
