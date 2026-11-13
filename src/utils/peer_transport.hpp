#ifndef UTILS_PEER_TRANSPORT_HPP_
#define UTILS_PEER_TRANSPORT_HPP_

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

#include "parthenon_mpi.hpp"
#include "utils/error_checking.hpp"
#include "utils/mpi_types.hpp"

namespace parthenon::peer_transport {

inline int CheckedMpiCount(const std::uint64_t count) {
  PARTHENON_REQUIRE_THROWS(
      count <= static_cast<std::uint64_t>(std::numeric_limits<int>::max()),
      "Peer transport count exceeds MPI INT_MAX");
  return static_cast<int>(count);
}

#ifdef MPI_PARALLEL
inline void ValidateTag(const mpi_comm_t comm, const int tag) {
  int *upper_bound = nullptr;
  int present = 0;
  PARTHENON_MPI_CHECK(MPI_Comm_get_attr(comm, MPI_TAG_UB, &upper_bound, &present));
  PARTHENON_REQUIRE_THROWS(present && upper_bound != nullptr,
                           "MPI communicator does not expose MPI_TAG_UB");
  PARTHENON_REQUIRE_THROWS(tag >= 0 && tag <= *upper_bound,
                           "Peer transport tag exceeds MPI_TAG_UB");
}

template <typename T>
void PostReceives(T *data, const std::uint64_t count, const int source, const int tag,
                  const mpi_comm_t comm, std::vector<mpi_request_t> &requests) {
  ValidateTag(comm, tag);
  std::uint64_t offset = 0;
  while (offset < count) {
    const auto chunk = std::min<std::uint64_t>(
        count - offset, static_cast<std::uint64_t>(std::numeric_limits<int>::max()));
    requests.emplace_back();
    PARTHENON_MPI_CHECK(MPI_Irecv(data + offset, CheckedMpiCount(chunk),
                                  MPITypeMap<T>::type(), source, tag, comm,
                                  &requests.back()));
    offset += chunk;
  }
}

template <typename T>
void PostSends(const T *data, const std::uint64_t count, const int destination,
               const int tag, const mpi_comm_t comm,
               std::vector<mpi_request_t> &requests) {
  ValidateTag(comm, tag);
  std::uint64_t offset = 0;
  while (offset < count) {
    const auto chunk = std::min<std::uint64_t>(
        count - offset, static_cast<std::uint64_t>(std::numeric_limits<int>::max()));
    requests.emplace_back();
    PARTHENON_MPI_CHECK(MPI_Isend(data + offset, CheckedMpiCount(chunk),
                                  MPITypeMap<T>::type(), destination, tag, comm,
                                  &requests.back()));
    offset += chunk;
  }
}
#endif

} // namespace parthenon::peer_transport

#endif // UTILS_PEER_TRANSPORT_HPP_
