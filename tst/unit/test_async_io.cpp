//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2026 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// This file was made in part with generative AI.

#include <atomic>
#include <future>
#include <stdexcept>
#include <vector>

#include <catch2/catch.hpp>

#include "outputs/async_io.hpp"

using parthenon::AsyncIOQueue;

TEST_CASE("AsyncIOQueue executes tasks in order", "[AsyncIOQueue]") {
  std::vector<int> order; // only accessed by the worker until futures are ready
  std::vector<std::future<void>> futures;
  {
    AsyncIOQueue queue;
    for (int i = 0; i < 20; i++) {
      futures.push_back(queue.Submit([&order, i] { order.push_back(i); }));
    }
    futures.back().get();
    REQUIRE(order.size() == 20);
    for (int i = 0; i < 20; i++) {
      REQUIRE(order[i] == i);
    }
  }
}

TEST_CASE("AsyncIOQueue propagates exceptions and keeps working", "[AsyncIOQueue]") {
  AsyncIOQueue queue;
  auto failing = queue.Submit([] { throw std::runtime_error("write failed"); });
  std::atomic<bool> ran{false};
  auto next = queue.Submit([&ran] { ran = true; });
  REQUIRE_THROWS_AS(failing.get(), std::runtime_error);
  next.get();
  REQUIRE(ran);
}

TEST_CASE("AsyncIOQueue destructor drains pending tasks", "[AsyncIOQueue]") {
  std::atomic<int> count{0};
  {
    AsyncIOQueue queue;
    for (int i = 0; i < 10; i++) {
      queue.Submit([&count] { count++; });
    }
  }
  REQUIRE(count == 10);
}
