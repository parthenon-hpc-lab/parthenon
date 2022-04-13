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

#include <type_traits>
#include <iostream>
#include <memory>
#include <utility>
#include <unordered_map>

#ifdef MPI_PARALLEL
#include <mpi.h>

#define request_t MPI_Request
#define comm_t MPI_Comm
#else
#define request_t int
#define comm_t int
#endif

namespace parthenon { 
namespace detail
{

#ifdef MPI_PARALLEL
  // MPIType<A> -> MPIType<A, bool_t<true>> and then
  // specialization checks are performed. This is why the
  // default parameter is required.
  template <bool>
  struct bool_t
  {
  };

  template <class T, class = bool_t<true>>
  struct MPIType;

  template <class U, class T>
  using MPI_type_check = std::is_same<U, std::remove_pointer_t<T>>;

  template <class T>
  struct MPIType<T, bool_t<MPI_type_check<double, T>::value>>
  {
    MPI_Datatype static value() noexcept { return MPI_DOUBLE; }
  };

  template <class T>
  struct MPIType<T, bool_t<MPI_type_check<Real, T>::value>>
  {
    MPI_Datatype static value() noexcept { return MPI_PARTHENON_REAL; }
  };

  template <class T>
  struct MPIType<T, bool_t<MPI_type_check<int, T>::value>>
  {
    MPI_Datatype static value() noexcept { return MPI_INT; }
  };

  template <class T>
  struct MPIType<T, bool_t<MPI_type_check<bool, T>::value>>
  {
    MPI_Datatype static value() noexcept { return MPI_CXX_BOOL; }
  };
#endif
}

using namespace detail;

class VariableCommMap
{
public:
  // This needs to be called in the same order on every process
  // TODO: Come up with a more robust scheme for creating variable communicators
  void AddVariable(const std::string &var_name)
  {
    if (map_.count(var_name) > 0)
      return; // Variable is already in map
#ifdef MPI_PARALLEL
    comm_t temp;
    MPI_Comm_dup(MPI_COMM_WORLD, &temp);
    map_[var_name] = temp;
#else
    map_[var_name] = current_idx_;
    ++current_idx_;
#endif
    return;
  }

  comm_t GetCommunicator(const std::string &var_name)
  {
    return map_[var_name];
  }

private:
  comm_t current_idx_; // Only needed when there is no MPI
  std::unordered_map<std::string, comm_t> map_;
};

enum class BufferState
{
  stale,
  sending,
  sending_null,
  receiving,
  received,
  received_null
};

enum class BuffCommType
{
  sender,
  receiver,
  both
};

template <class T>
class CommBuffer : public T
{

  // Need specializations to be friends with each other
  template <typename U>
  friend class CommBuffer;

  std::shared_ptr<BufferState> state_;
  std::shared_ptr<BuffCommType> comm_type_;
  std::shared_ptr<bool> recv_start_called_;
  std::shared_ptr<request_t> my_request_;

  int my_rank;
  int tag_;
  int send_rank_;
  int recv_rank_;
  comm_t comm_;

  using buf_base_t = std::remove_pointer_t<decltype(std::declval<T>().data())>;
  buf_base_t null_buf_;
  bool active_ = false;

  std::function<T()> get_resource_;

public:

  CommBuffer()
#ifdef MPI_PARALLEL
      : my_request_(std::make_shared<MPI_Request>(MPI_REQUEST_NULL))
#endif
  {
  }

  CommBuffer(int tag, int send_rank, int recv_rank, comm_t comm_, std::function<T()> get_resource);

  template <class U>
  CommBuffer(const CommBuffer<U> &in);

  template <class U>
  CommBuffer &operator=(const CommBuffer<U> &in);

  void Allocate()
  {
    T::operator=(get_resource_());
    active_ = true;
  }

  void Free()
  {
    T::operator=(T());
    active_ = false;
  }

  bool IsActive() const { return active_; }

  void Send() noexcept;
  void SendNull() noexcept;

  bool TryReceive() noexcept;

  void IsSentGuard() noexcept
  {
#ifdef MPI_PARALLEL
    if (*comm_type_ == BuffCommType::sender)
    {
      MPI_Wait(my_request_.get(), MPI_STATUS_IGNORE);
    }
#endif
  }

  void Stale() noexcept
  {
    *state_ = BufferState::stale;
  }
};

// Definitions below

template <class T>
CommBuffer<T>::CommBuffer(int tag, int send_rank, int recv_rank, comm_t comm, std::function<T()> get_resource)
    : state_(std::make_shared<BufferState>(BufferState::stale)),
      comm_type_(std::make_shared<BuffCommType>(BuffCommType::both)),
      recv_start_called_(std::make_shared<bool>(false)),
#ifdef MPI_PARALLEL
      my_request_(std::make_shared<MPI_Request>(MPI_REQUEST_NULL)),
#endif
      tag_(tag),
      send_rank_(send_rank),
      recv_rank_(recv_rank),
      comm_(comm),
      get_resource_(get_resource),
      T()
{

// Set up persistent communication
#ifdef MPI_PARALLEL
  MPI_Comm_rank(comm_, &my_rank);
#else
  my_rank = 0;
#endif
  if (send_rank == recv_rank)
  {
    assert(my_rank == send_rank);
    *comm_type_ = BuffCommType::both;
  }
  else if (my_rank == send_rank)
  {
    *comm_type_ = BuffCommType::sender;
  }
  else if (my_rank == recv_rank)
  {
    *comm_type_ = BuffCommType::receiver;
  }
  else
  {
    // This is an error
    std::cout << "CommBuffer initialization error" << std::endl;
  }
}

template <class T>
template <class U>
CommBuffer<T>::CommBuffer(const CommBuffer<U> &in) : T(in),
                                                     state_(in.state_),
                                                     comm_type_(in.comm_type_),
                                                     recv_start_called_(in.recv_start_called_),
                                                     my_request_(in.my_request_),
                                                     tag_(in.tag_),
                                                     send_rank_(in.send_rank_),
                                                     recv_rank_(in.recv_rank_),
                                                     comm_(in.comm_),
                                                     active_(in.active_)
{
#ifdef MPI_PARALLEL
  MPI_Comm_rank(comm_, &my_rank);
#else
  my_rank = 0;
#endif
}

template <class T>
template <class U>
CommBuffer<T> &CommBuffer<T>::operator=(const CommBuffer<U> &in)
{
  T::operator=(static_cast<U>(in));
  state_ = in.state_;
  comm_type_ = in.comm_type_;
  recv_start_called_ = in.recv_start_called_;
  my_request_ = in.my_request_;
  tag_ = in.tag_;
  send_rank_ = in.send_rank_;
  recv_rank_ = in.recv_rank_;
  comm_ = in.comm_;
  active_ = in.active_;
#ifdef MPI_PARALLEL
  MPI_Comm_rank(comm_, &my_rank);
#else
  my_rank = 0;
#endif
  return *this;
}

template <class T>
void CommBuffer<T>::Send() noexcept
{
  if (!active_)
  {
    SendNull();
    return;
  }
  if (*comm_type_ == BuffCommType::sender)
  {
// Make sure that this request isn't still out,
// this could be blocking
#ifdef MPI_PARALLEL
    MPI_Wait(my_request_.get(), MPI_STATUS_IGNORE);
    MPI_Isend(T::data(), T::size(), MPIType<buf_base_t>::value(),
              recv_rank_, tag_, comm_, my_request_.get());
#endif
  }
  *state_ = BufferState::sending;
  if (*comm_type_ == BuffCommType::receiver)
  {
    // This is an error
    Kokkos::abort("Trying to send from a receiver");
  }
}

template <class T>
void CommBuffer<T>::SendNull() noexcept
{
  if (*comm_type_ == BuffCommType::sender)
  {
// Make sure that this request isn't still out,
// this could be blocking
#ifdef MPI_PARALLEL
    MPI_Wait(my_request_.get(), MPI_STATUS_IGNORE);
    MPI_Isend(&null_buf_, 0, MPIType<buf_base_t>::value(),
              recv_rank_, tag_, comm_, my_request_.get());
#endif
  }
  *state_ = BufferState::sending_null;
  if (*comm_type_ == BuffCommType::receiver)
  {
    // This is an error
    Kokkos::abort("Trying to send from a receiver");
  }
}

template <class T>
bool CommBuffer<T>::TryReceive() noexcept
{
  if (*state_ == BufferState::received || *state_ == BufferState::received_null)
    return true;
  if (*comm_type_ == BuffCommType::receiver)
  {
    if (!*recv_start_called_)
    {
      *recv_start_called_ = true;
    }
#ifdef MPI_PARALLEL
    int test;
    MPI_Status status;
    MPI_Iprobe(send_rank_, tag_, comm_, &test, &status);

    if (test)
    {
      int size;
      MPI_Get_count(&status, MPIType<buf_base_t>::value(), &size);
      if (size > 0)
      {
        if (!active_)
          Allocate();
        MPI_Recv(T::data(), T::size(), MPIType<buf_base_t>::value(), send_rank_,
                 tag_, comm_, MPI_STATUS_IGNORE);
        *state_ = BufferState::received;
      }
      else
      {
        if (active_)
          Free();
        MPI_Recv(&null_buf_, 0, MPIType<buf_base_t>::value(), send_rank_,
                 tag_, comm_, MPI_STATUS_IGNORE);
        *state_ = BufferState::received_null;
      }
      *recv_start_called_ = false;
      return true;
    }
    return false;
#else
    return true;
#endif
  }
  else if (*comm_type_ == BuffCommType::both)
  {
    if (*state_ == BufferState::sending)
    {
      *state_ = BufferState::received;
      // Memory should already be available, since both
      // send and receive rank point at the same memory
      return true;
    }
    else if (*state_ == BufferState::sending_null)
    {
      *state_ = BufferState::received_null;
      return true;
    }
    return false;
  }
  else
  {
    // This is an error since this is a purely send buffer
    Kokkos::abort("Trying to receive on a sender");
  }
  return false;
}
} // namespace parthenon 
#endif // UTILS_COMMUNICATION_BUFFER_HPP_