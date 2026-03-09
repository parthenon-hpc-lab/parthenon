//========================================================================================
// (C) (or copyright) 2020-2026. Triad National Security, LLC. All rights reserved.
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

#include <iostream>
#include <istream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <catch2/catch.hpp>

#include "basic_types.hpp"
#include "globals.hpp"
#include "parthenon_mpi.hpp"
#include "utils/string_utils.hpp"

using parthenon::Real;
constexpr Real EPS = 1e-12;

SCENARIO("We can parse a simple ASCII table", "[AsciiTableParser][StringUtils]") {
  GIVEN("A gross but valid ascii table") {
    std::stringstream ss;
    ss << "# header comment line\n"
       << "                # weird header comment line\n"
       << "\n" // empty line
       << "0\t0.0\t123.0e4\n"
       << "1 2.0 \t -4567.89e-1     " << std::endl // trailing whitespace
       << "\t     3 4.0 5   \t  # inine comment" << std::endl
       << std::endl;
    WHEN("We parse it") {
      std::istringstream s(ss.str());
      auto table = parthenon::string_utils::ParseAsciiTable<Real>(s);
      THEN("The resultant table has the right number of rows and columns") {
        REQUIRE(table.rows == 3);
        REQUIRE(table.cols == 3);
        AND_THEN("The resultant table as the correct contents") {
          REQUIRE(table(0, 0) == 0.0);
          REQUIRE(table(0, 1) == 0.0);
          REQUIRE(std::abs(table(0, 2) - 123e4) < EPS);
          REQUIRE(table(1, 0) == 1);
          REQUIRE(table(1, 1) == 2);
          REQUIRE(std::abs(table(1, 2) - -4567.89e-1) < EPS);
          REQUIRE(table(2, 0) == 3);
          REQUIRE(table(2, 1) == 4);
          REQUIRE(table(2, 2) == 5);
        }
      }
    }
  }

  GIVEN("A table containing ints with only 1 row") {
    std::stringstream ss;
    ss << "1 2 3 4 5" << std::endl;
    WHEN("We parse it") {
      std::istringstream s(ss.str());
      auto table = parthenon::string_utils::ParseAsciiTable<int>(s);
      THEN("The resultant table has the right number of rows and columns") {
        REQUIRE(table.rows == 1);
        REQUIRE(table.cols == 5);
        AND_THEN("The contents are correct") {
          for (int i = 1; i < 5; ++i) {
            REQUIRE(table(0, i - 1) == i);
          }
        }
      }
    }
  }

  GIVEN("An empty ascii table") {
    std::stringstream ss;
    ss << "# some header" << std::endl
       << "# but no actual contents" << std::endl
       << std::endl;
    WHEN("When we attempt to parse it") {
      std::istringstream s(ss.str());
      auto table = parthenon::string_utils::ParseAsciiTable<Real>(s);
      THEN("We get an empty table object") { REQUIRE(table.data.size() == 0); }
    }
  }

  GIVEN("A ragged table") {
    std::stringstream ss;
    ss << "1 2 3 4\n"
       << "5 6\n"
       << "7 8 9 10" << std::endl;
    WHEN("We attempt to parse it") {
      std::istringstream s(ss.str());
      THEN("Parthenon throws an error") {
        REQUIRE_THROWS_AS(parthenon::string_utils::ParseAsciiTable<int>(s),
                          std::runtime_error);
      }
    }
  }
}

SCENARIO("We can MPI broadcast a string from a file",
         "[MPI][BroadcastFileString][StringUtils]") {
  GIVEN("A file that contains a simple string") {
    std::stringstream ss;
    ss << "# header comment line\n"
       << "0 0.0 123.0e4\n"
       << "1 2.0 -4567.89e-1\n"
       << "3 4.0 5.0\n"
       << std::endl;
    std::string teststring = ss.str();

    const std::string filename = "testfile.txt";
    if (parthenon::Globals::my_rank == 0) {
      std::ofstream out(filename);
      out << teststring;
    }

    WHEN("We try to read the file via broadcast") {
      auto newstring = parthenon::string_utils::BroadcastFileString(filename);
      THEN("The strings match") { REQUIRE(newstring == teststring); }
    }

    WHEN("We parse it via broadcast") {
      auto table = parthenon::string_utils::ParseAsciiTable<Real>(filename);
      THEN("The resultant table has the right number of rows and columns") {
        REQUIRE(table.rows == 3);
        REQUIRE(table.cols == 3);
        AND_THEN("The resultant table as the correct contents") {
          REQUIRE(table(0, 0) == 0.0);
          REQUIRE(table(0, 1) == 0.0);
          REQUIRE(std::abs(table(0, 2) - 123e4) < EPS);
          REQUIRE(table(1, 0) == 1);
          REQUIRE(table(1, 1) == 2);
          REQUIRE(std::abs(table(1, 2) - -4567.89e-1) < EPS);
          REQUIRE(table(2, 0) == 3);
          REQUIRE(table(2, 1) == 4);
          REQUIRE(table(2, 2) == 5);
        }
      }
    }
  }
}
