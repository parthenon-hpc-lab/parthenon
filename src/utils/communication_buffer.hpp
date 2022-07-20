//========================================================================================
// (C) (or copyright) 2020-2022. Triad National Security, LLC. All rights reserved.
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

#ifndef UTILS_COMMUNICATION_BUFFER_HPP_
#define UTILS_COMMUNICATION_BUFFER_HPP_

#include <iostream>
#include <limits>
#include <memory>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include "utils/mpi_types.hpp"

#ifdef MPI_PARALLEL
#include <mpi.h>

#define request_t MPI_Request
#define comm_t MPI_Comm
#else
#define request_t int
#define comm_t int
#endif

namespace parthenon {

//             Read    Write
//    stale:             X
// sending*:
// received:     X
enum class BufferState { stale, sending, sending_null, received, received_null };

enum class BuffCommType { sender, receiver, both, sparse_receiver };

enum class BoundaryType : int { local, nonlocal, any, reflux_send, reflux_recv };

template <class T>
class CommBuffer {
 private:
  // Need specializations to be friends with each other
  template <typename U>
  friend class CommBuffer;

  std::shared_ptr<BufferState> state_;
  std::shared_ptr<BuffCommType> comm_type_;
  std::shared_ptr<bool> started_irecv_;
  std::shared_ptr<int> nrecv_tries_;
  std::shared_ptr<request_t> my_request_;

  int my_rank;
  int tag_;
  int send_rank_;
  int recv_rank_;
  comm_t comm_;

  using buf_base_t = std::remove_pointer_t<decltype(std::declval<T>().data())>;
  buf_base_t null_buf_ = std::numeric_limits<buf_base_t>::signaling_NaN();
  bool active_ = false;

  std::function<T()> get_resource_;

  T buf_;

 public:
  CommBuffer()
#ifdef MPI_PARALLEL
      : my_request_(std::make_shared<MPI_Request>(MPI_REQUEST_NULL))
#endif
  {
  }

  CommBuffer(int tag, int send_rank, int recv_rank, comm_t comm_,
             std::function<T()> get_resource, bool do_sparse_allocation = false);

  ~CommBuffer();

  template <class U>
  CommBuffer(const CommBuffer<U> &in);

  template <class U>
  CommBuffer &operator=(const CommBuffer<U> &in);

  operator T &() { return buf_; }
  operator const T &() const { return buf_; }

  T &buffer() { return buf_; }
  const T &buffer() const { return buf_; }

  void Allocate() {
    if (!active_) {
      buf_ = get_resource_();
      active_ = true;
    }
  }

  void Free() {
    buf_ = T();
    active_ = false;
  }

  bool IsActive() const { return active_; }

  BufferState GetState() { return *state_; }

  void Send() noexcept;
  void SendNull() noexcept;

  bool IsAvailableForWrite();

  void TryStartReceive() noexcept;
  bool TryReceive() noexcept;

  void Stale();
};

// Method definitions below

template <class T>
CommBuffer<T>::CommBuffer(int tag, int send_rank, int recv_rank, comm_t comm,
                          std::function<T()> get_resource, bool do_sparse_allocation)
    : state_(std::make_shared<BufferState>(BufferState::stale)),
      comm_type_(std::make_shared<BuffCommType>(BuffCommType::both)),
      started_irecv_(std::make_shared<bool>(false)),
      nrecv_tries_(std::make_shared<int>(0)),
#ifdef MPI_PARALLEL
      my_request_(std::make_shared<MPI_Request>(MPI_REQUEST_NULL)),
#endif
      tag_(tag), send_rank_(send_rank), recv_rank_(recv_rank), comm_(comm),
      get_resource_(get_resource), buf_() {
// Set up persistent communication
#ifdef MPI_PARALLEL
  PARTHENON_MPI_CHECK(MPI_Comm_rank(comm_, &my_rank));
#else
  my_rank = 0;
#endif
  if (send_rank == recv_rank) {
    assert(my_rank == send_rank);
    *comm_type_ = BuffCommType::both;
  } else if (my_rank == send_rank) {
    *comm_type_ = BuffCommType::sender;
  } else if (my_rank == recv_rank) {
    *comm_type_ = BuffCommType::receiver;
    if (do_sparse_allocation) *comm_type_ = BuffCommType::sparse_receiver;
  } else {
    // This is an error
    std::cout << "CommBuffer initialization error" << std::endl;
  }
}

template <class T>
template <class U>
CommBuffer<T>::CommBuffer(const CommBuffer<U> &in)
    : buf_(in.buf_), state_(in.state_), comm_type_(in.comm_type_),
      started_irecv_(in.started_irecv_), nrecv_tries_(in.nrecv_tries_),
      my_request_(in.my_request_), tag_(in.tag_), send_rank_(in.send_rank_),
      recv_rank_(in.recv_rank_), comm_(in.comm_), active_(in.active_) {
#ifdef MPI_PARALLEL
  PARTHENON_MPI_CHECK(MPI_Comm_rank(comm_, &my_rank));
#else
  my_rank = 0;
#endif
}

template <class T>
CommBuffer<T>::~CommBuffer() {
#ifdef MPI_PARALLEL
  if (my_request_.use_count() == 1) { // This is the last shallow copy of this buffer
    int flag;
    MPI_Status status;
    PARTHENON_MPI_CHECK(MPI_Test(my_request_.get(), &flag, &status));
    if (!flag) {
      if (*comm_type_ == BuffCommType::sender) {
        PARTHENON_MPI_CHECK(MPI_Wait(my_request_.get(), MPI_STATUS_IGNORE));
      } else {
        PARTHENON_MPI_CHECK(MPI_Cancel(my_request_.get()));
        PARTHENON_MPI_CHECK(MPI_Wait(my_request_.get(), MPI_STATUS_IGNORE));
      }
    }
  }
#endif
}

template <class T>
template <class U>
CommBuffer<T> &CommBuffer<T>::operator=(const CommBuffer<U> &in) {
  buf_ = in.buf_;
  state_ = in.state_;
  comm_type_ = in.comm_type_;
  started_irecv_ = in.started_irecv_;
  nrecv_tries_ = in.nrecv_tries_;
  my_request_ = in.my_request_;
  tag_ = in.tag_;
  send_rank_ = in.send_rank_;
  recv_rank_ = in.recv_rank_;
  comm_ = in.comm_;
  active_ = in.active_;
#ifdef MPI_PARALLEL
  PARTHENON_MPI_CHECK(MPI_Comm_rank(comm_, &my_rank));
#else
  my_rank = 0;
#endif
  return *this;
}

template <class T>
void CommBuffer<T>::Send() noexcept {
  if (!active_) {
    SendNull();
    return;
  }

  PARTHENON_DEBUG_REQUIRE(*state_ == BufferState::stale,
                          "Trying to send from buffer that hasn't been staled.");
  *state_ = BufferState::sending;
  if (*comm_type_ == BuffCommType::sender) {
// Make sure that this request isn't still out,
// this could be blocking
#ifdef MPI_PARALLEL
    PARTHENON_REQUIRE(
        buf_.size() > 0,
        "Trying to send zero size buffer, which will be interpreted as sending_null.");
    PARTHENON_MPI_CHECK(MPI_Wait(my_request_.get(), MPI_STATUS_IGNORE));
    PARTHENON_MPI_CHECK(MPI_Isend(buf_.data(), buf_.size(),
                                  MPITypeMap<buf_base_t>::type(), recv_rank_, tag_, comm_,
                                  my_request_.get()));
#endif
  }
  if (*comm_type_ == BuffCommType::receiver) {
    // This is an error
    Kokkos::abort("Trying to send from a receiver");
  }
}

template <class T>
void CommBuffer<T>::SendNull() noexcept {
  PARTHENON_DEBUG_REQUIRE(*state_ == BufferState::stale,
                          "Trying to send_null from buffer that hasn't been staled.");
  *state_ = BufferState::sending_null;
  if (*comm_type_ == BuffCommType::sender) {
// Make sure that this request isn't still out,
// this could be blocking
#ifdef MPI_PARALLEL
    PARTHENON_MPI_CHECK(MPI_Wait(my_request_.get(), MPI_STATUS_IGNORE));
    PARTHENON_MPI_CHECK(MPI_Isend(&null_buf_, 0, MPITypeMap<buf_base_t>::type(),
                                  recv_rank_, tag_, comm_, my_request_.get()));
#endif
  }
  if (*comm_type_ == BuffCommType::receiver) {
    // This is an error
    Kokkos::abort("Trying to send from a receiver");
  }
}

template <class T>
bool CommBuffer<T>::IsAvailableForWrite() {
  if (*comm_type_ == BuffCommType::sender) {
#ifdef MPI_PARALLEL
    // We do not check stale status here since the receiving end should be the one
    // setting the buffer to stale, all we care about for a pure sender is wether
    // or not its last send message has been completed
    if (*state_ == BufferState::stale) return true;
    if (*my_request_ == MPI_REQUEST_NULL) return true;
    int flag, test;
    PARTHENON_MPI_CHECK(MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &test,
                                   MPI_STATUS_IGNORE));
    PARTHENON_MPI_CHECK(MPI_Test(my_request_.get(), &flag, MPI_STATUS_IGNORE));
    if (flag) *state_ = BufferState::stale;
    return flag;
#else
    PARTHENON_FAIL("Should not have a sending buffer when MPI is not enabled.");
#endif
  } else if (*comm_type_ == BuffCommType::both) {
    if (*state_ == BufferState::stale) return true;
    return false;
  } else {
    PARTHENON_FAIL("Receiving buffer is never available for write.");
  }
}

template <class T>
void CommBuffer<T>::TryStartReceive() noexcept {
#ifdef MPI_PARALLEL
  if (*comm_type_ == BuffCommType::receiver && !*started_irecv_) {
    PARTHENON_REQUIRE(
        *my_request_ == MPI_REQUEST_NULL,
        "Cannot have another pending request in a buffer that is starting to receive.");
    if (!IsActive())
      Allocate(); // For early start of Irecv, always need storage space even if not used
    PARTHENON_MPI_CHECK(MPI_Irecv(buf_.data(), buf_.size(),
                                  MPITypeMap<buf_base_t>::type(), send_rank_, tag_, comm_,
                                  my_request_.get()));
    *started_irecv_ = true;
  } else if (*comm_type_ == BuffCommType::sparse_receiver && !*started_irecv_) {
    int test;
    MPI_Status status;
    // This is the extra MPI call that impacts performance mentioned in Athena++
    // PARTHENON_MPI_CHECK(MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &test,
    //                               MPI_STATUS_IGNORE));
    // Check if our message is available so that we can use the correct buffer size
    PARTHENON_MPI_CHECK(MPI_Iprobe(send_rank_, tag_, comm_, &test, &status));
    if (test) {
      // For optimal buffer memory useage, we can only post the Irecv once we know the
      // size of the incoming buffer
      int size;
      PARTHENON_MPI_CHECK(MPI_Get_count(&status, MPITypeMap<buf_base_t>::type(), &size));
      if (size > 0) {
        if (!active_) Allocate();
        PARTHENON_MPI_CHECK(MPI_Irecv(buf_.data(), buf_.size(),
                                      MPITypeMap<buf_base_t>::type(), send_rank_, tag_,
                                      comm_, my_request_.get()));
      } else {
        if (active_) Free();
        PARTHENON_MPI_CHECK(MPI_Irecv(&null_buf_, 0, MPITypeMap<buf_base_t>::type(),
                                      send_rank_, tag_, comm_, my_request_.get()));
      }
      *started_irecv_ = true;
    }
  }
#endif
}

template <class T>
bool CommBuffer<T>::TryReceive() noexcept {
  if (*state_ == BufferState::received || *state_ == BufferState::received_null)
    return true;

  if (*comm_type_ == BuffCommType::receiver ||
      *comm_type_ == BuffCommType::sparse_receiver) {
#ifdef MPI_PARALLEL
    (*nrecv_tries_)++;
    if (*nrecv_tries_ > 1e6)
      PARTHENON_FAIL("MPI probably hanging after 1e6 receive tries.");

    TryStartReceive();

    if (*started_irecv_) {
      MPI_Status status;
      int flag;
      // This is the extra MPI call that impacts performance as mentioned in Athena++.
      // Depending on the MPI implementation and the order of MPI_Test calls, as well
      // as the total number of buffers being communicated, I have found this can have
      // anywhere from no impact on the walltime to a factor of a few reduction in the
      // walltime. It seems to be unpredictable as far as I can tell.
      for (int i = 0; i < 1; ++i)
        PARTHENON_MPI_CHECK(MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag,
                                       MPI_STATUS_IGNORE));
      PARTHENON_MPI_CHECK(MPI_Test(my_request_.get(), &flag, &status));
      if (flag) {
        // Check the size of the message, it will be zero if the sender wants you to use
        // default buffer data
        int size;
        PARTHENON_MPI_CHECK(
            MPI_Get_count(&status, MPITypeMap<buf_base_t>::type(), &size));

        PARTHENON_REQUIRE(*my_request_ == MPI_REQUEST_NULL,
                          "MPI request should be finished to get here.");
        // Set flags based on a finished receive
        *started_irecv_ = false;
        *nrecv_tries_ = 0;
        if (size > 0)
          *state_ = BufferState::received;
        else
          *state_ = BufferState::received_null;

        return true;
      }
    }
    return false;
#else
    PARTHENON_FAIL("Should not have a purely receiving buffer without MPI enabled.");
    return false;
#endif
  } else if (*comm_type_ == BuffCommType::both) {
    if (*state_ == BufferState::sending) {
      *state_ = BufferState::received;
      // Memory should already be available, since both
      // send and receive rank point at the same memory
      return true;
    } else if (*state_ == BufferState::sending_null) {
      *state_ = BufferState::received_null;
      return true;
    }
    return false;
  } else {
    // This is an error since this is a purely send buffer
    PARTHENON_FAIL("Trying to receive on a sender");
  }
  return false;
}

template <class T>
void CommBuffer<T>::Stale() {
  if (*comm_type_ == BuffCommType::sender) PARTHENON_FAIL("Should never get here.");

  if (!(*state_ == BufferState::received || *state_ == BufferState::received_null))
    PARTHENON_DEBUG_WARN("Staling buffer not in the received state.");
#ifdef MPI_PARALLEL
  if (MPI_REQUEST_NULL != *my_request_)
    PARTHENON_WARN("Staling buffer with pending request.");
#endif
  *state_ = BufferState::stale;
}

} // namespace parthenon
#endif // UTILS_COMMUNICATION_BUFFER_HPP_
