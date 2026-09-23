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

#ifndef TENSORS_TT_COMM_CHANNEL_HPP
#define TENSORS_TT_COMM_CHANNEL_HPP

#include <memory>

#include "tensors/tt_types.hpp"
#include "utils/communication_buffer.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {

// Same-rank tensor-train communication channel. Like CommBuffer it tracks a
// stale -> sending -> received handshake, but it carries a whole TensorTrain (dynamic
// ranks) rather than a fixed-size buffer: the payload is a variable-size train, so there
// is no persistent fixed buffer the way regular-field comm has.
class TTCommChannel {
 public:
  using train_t = tensor::TensorTrain;
  using train_ptr = std::shared_ptr<train_t>;

  TTCommChannel() = default;

  BufferState GetState() const { return state_; }

  bool IsAvailableForWrite() const { return state_ == BufferState::stale; }

  // Deposit the addend train into the channel and mark it sent.
  void Send(train_ptr train) {
    PARTHENON_DEBUG_REQUIRE(state_ == BufferState::stale,
                            "Trying to send on a channel that hasn't been staled.");
    train_ = std::move(train);
    state_ = BufferState::sending;
  }

  // Transition to received. Returns false if the sender has not deposited yet.
  bool TryReceive() {
    if (state_ == BufferState::received) return true;
    if (state_ == BufferState::sending) {
      state_ = BufferState::received;
      return true;
    }
    return false;
  }

  // The shipped train (valid once received).
  train_ptr Get() const { return train_; }

  // Release the payload and return the channel to a writable state.
  void Stale() {
    train_ = nullptr;
    state_ = BufferState::stale;
  }

 private:
  BufferState state_ = BufferState::stale;
  train_ptr train_;
};

} // namespace parthenon

#endif // TENSORS_TT_COMM_CHANNEL_HPP
