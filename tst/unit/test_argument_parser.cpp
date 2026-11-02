// This file was made in part with generative AI.

//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2026 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
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

#include <array>
#include <string>

#include <catch2/catch.hpp>

#include "argument_parser.hpp"

using parthenon::ArgParse;
using parthenon::ArgStatus;

TEST_CASE("Analysis command-line parsing", "[ArgParse][Analysis]") {
  SECTION("Analysis requires and records an input deck") {
    std::array<char, 4> executable{"app"};
    std::array<char, 3> analysis_option{"-a"};
    std::array<char, 10> dump{"dump.phdf"};
    std::array<char, 3> input_option{"-i"};
    std::array<char, 10> input{"input.pin"};
    char *argv[] = {executable.data(), analysis_option.data(), dump.data(),
                    input_option.data(), input.data()};

    ArgParse arguments;
    REQUIRE(arguments.parse(5, argv) == ArgStatus::ok);
    REQUIRE(arguments.analysis_flag);
    REQUIRE_FALSE(arguments.is_restart);
    REQUIRE(std::string(arguments.restart_filename) == "dump.phdf");
    REQUIRE(std::string(arguments.input_filename) == "input.pin");
  }

  SECTION("Analysis without an input deck is rejected") {
    std::array<char, 4> executable{"app"};
    std::array<char, 3> analysis_option{"-a"};
    std::array<char, 8> dump{"dump.bp"};
    char *argv[] = {executable.data(), analysis_option.data(), dump.data()};

    ArgParse arguments;
    REQUIRE(arguments.parse(3, argv) == ArgStatus::error);
  }

  SECTION("Analysis and restart modes cannot be combined") {
    std::array<char, 4> executable{"app"};
    std::array<char, 3> analysis_option{"-a"};
    std::array<char, 10> dump{"dump.phdf"};
    std::array<char, 3> restart_option{"-r"};
    std::array<char, 13> restart{"restart.rhdf"};
    std::array<char, 3> input_option{"-i"};
    std::array<char, 10> input{"input.pin"};
    char *argv[] = {executable.data(),     analysis_option.data(), dump.data(),
                    restart_option.data(), restart.data(),         input_option.data(),
                    input.data()};

    ArgParse arguments;
    REQUIRE(arguments.parse(7, argv) == ArgStatus::error);
  }
}
