#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

namespace plb {

struct ChunkRange {
  std::size_t start = 0;
  std::size_t length = 0;
};

inline std::vector<ChunkRange> BuildChunks(std::size_t total, std::size_t chunk_length) {
  std::vector<ChunkRange> chunks;
  if (total == 0) {
    return chunks;
  }
  const std::size_t safe_chunk = std::max<std::size_t>(1, chunk_length);
  chunks.reserve((total + safe_chunk - 1) / safe_chunk);
  for (std::size_t start = 0; start < total; start += safe_chunk) {
    chunks.push_back({start, std::min(safe_chunk, total - start)});
  }
  return chunks;
}

inline std::size_t DefaultHierarchicalChunkLength(int ni, int nj, int requested) {
  if (requested > 0) {
    return static_cast<std::size_t>(requested);
  }
  return static_cast<std::size_t>(std::max(1, ni * nj));
}

inline std::size_t DefaultTunedChunkLength(int ni, int requested) {
  if (requested > 0) {
    return static_cast<std::size_t>(requested);
  }
  return static_cast<std::size_t>(std::max(1, ni));
}

} // namespace plb
