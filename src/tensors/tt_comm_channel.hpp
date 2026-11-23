//========================================================================================
// (C) (or copyright) 2026. Triad National Security, LLC. All rights reserved.
//
// This program was produced under U.S. Government contract 89233218CNA000001 for Los
// Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
// for the U.S. Department of Energy/National Nuclear Security Administration. All rights
// in the program are reserved by Triad National Security, LLC, and the U.S. Department
// of Energy/National Nuclear Security Administration. The Government is granted for
// itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
// license in this material to reproduce, prepare derivative works, distribute copies to
// the public, perform publicly and display publicly, and to permit others to do so.
//========================================================================================

// This file was made in part with generative AI.

#ifndef TENSORS_TT_COMM_CHANNEL_HPP_
#define TENSORS_TT_COMM_CHANNEL_HPP_

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "kokkos_types.hpp"
#include "tensors/tt_types.hpp"
#include "utils/communication_buffer.hpp"
#include "utils/error_checking.hpp"
#include "utils/mpi_types.hpp"

namespace parthenon {

// Name of the MPI communicator used for a TT field's comm. Prefixed so that it cannot
// collide with a regular field of the same name in the mesh communicator map.
inline std::string TTCommLabel(const std::string &label) { return "tt::" + label; }

// Tensor-train communication channel. Like CommBuffer it tracks a
// stale -> sending -> received handshake, but it carries a whole TensorTrain (dynamic
// ranks) rather than a fixed-size buffer. A channel is one of:
//
//   local:  sender and receiver are on the same rank and share this object; the train is
//           handed over by shared_ptr, with no copy.
//   remote: sender and receiver hold separate channels on different ranks, each wrapping
//           a CommBuffer. The sender serializes the train into the buffer (see
//           tt_serialize.hpp) and sends it; the receiver is a sparse receiver, so it
//           probes the message size and grows its buffer to fit. Once received, the
//           buffer is deserialized and the train installed with SetTrain.
//
// The method names and states mirror CommBuffer so that remote channels can later be
// folded into coalesced communication alongside regular field buffers.
class TTCommChannel {
 public:
  using train_t = tensor::TensorTrain;
  using train_ptr = std::shared_ptr<train_t>;
  using real_t = typename train_t::traits::real_t;
  using buf_t = BufArray1D<real_t>;
  using comm_buf_t = CommBuffer<buf_t>;

  // A local channel.
  TTCommChannel() = default;

  // A remote channel between send_rank and recv_rank, one of which must be this rank.
  TTCommChannel(int tag, int send_rank, int recv_rank, mpi_comm_t comm)
      : buf_(
            std::in_place, tag, send_rank, recv_rank, comm,
            [](int n) { return buf_t("tt_comm_buf", n); },
            /*do_sparse_allocation=*/true) {}

  bool IsLocal() const { return !buf_.has_value(); }

  BufferState GetState() const { return IsLocal() ? state_ : buf_->GetState(); }

  bool IsAvailableForWrite() const {
    return IsLocal() ? state_ == BufferState::stale : buf_->IsAvailableForWrite();
  }

  // Local: deposit the addend train into the channel and mark it sent.
  void Send(train_ptr train) {
    PARTHENON_REQUIRE(IsLocal(), "Remote TT channels send a serialized buffer.");
    PARTHENON_DEBUG_REQUIRE(state_ == BufferState::stale,
                            "Trying to send on a channel that hasn't been staled.");
    train_ = std::move(train);
    state_ = BufferState::sending;
  }

  // Remote: make the send buffer hold at least size values and return it to be
  // serialized into. The buffer only grows, so it is reused across exchanges.
  buf_t &PrepareSendBuffer(std::size_t size) {
    PARTHENON_REQUIRE(!IsLocal(), "Local TT channels have no send buffer.");
    if (!buf_->IsActive() || buf_->buffer().size() < size)
      buf_->Allocate(static_cast<int>(size));
    return buf_->buffer();
  }

  // Remote: send the first size values of the send buffer.
  void SendBuffer(std::size_t size) {
    PARTHENON_REQUIRE(!IsLocal(), "Local TT channels send a train.");
    buf_->Send(false, static_cast<int>(size));
  }

  // Transition to received. Returns false if the train has not arrived yet.
  bool TryReceive() {
    if (!IsLocal()) return buf_->TryReceive();
    if (state_ == BufferState::received) return true;
    if (state_ == BufferState::sending) {
      state_ = BufferState::received;
      return true;
    }
    return false;
  }

  // Remote: the received serialized train (valid once received). The buffer may be
  // larger than the message; the serialized header gives the true length.
  const buf_t &ReceiveBuffer() const {
    PARTHENON_REQUIRE(!IsLocal(), "Local TT channels have no receive buffer.");
    return buf_->buffer();
  }

  // Remote: install the train deserialized from the receive buffer.
  void SetTrain(train_ptr train) { train_ = std::move(train); }

  // The shipped train (valid once received, and for remote channels once set).
  train_ptr Get() const { return train_; }

  // Release the payload and return the channel to a writable state.
  void Stale() {
    train_ = nullptr;
    if (IsLocal()) {
      state_ = BufferState::stale;
    } else {
      buf_->Stale();
    }
  }

  // Whether the channel can be destroyed without losing an in-flight message.
  bool IsSafeToDelete() const {
    return IsLocal() ? state_ == BufferState::stale : buf_->IsSafeToDelete();
  }

 private:
  BufferState state_ = BufferState::stale;
  train_ptr train_;
  // CommBuffer's state queries are not const-qualified, though they only read or
  // advance MPI progress.
  mutable std::optional<comm_buf_t> buf_;
};

} // namespace parthenon

#endif // TENSORS_TT_COMM_CHANNEL_HPP_
