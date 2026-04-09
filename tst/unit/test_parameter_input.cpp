//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020-2025. Triad National Security, LLC. All rights reserved.
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

#include "parameter_input.hpp"

using parthenon::ParameterInput;

TEST_CASE("Test required/desired checking from inputs", "[ParameterInput]") {
  GIVEN("A ParameterInput object already populated") {
    ParameterInput in;
    std::stringstream ss;
    ss << "<block1>" << std::endl
       << "var1 = 0   # comment" << std::endl
       << "var2 = 1,  & # another comment" << std::endl
       << "       2" << std::endl
       << "<block2>" << std::endl
       << "var3 = 3" << std::endl
       << "# comment" << std::endl
       << "var4 = 4" << std::endl
       << "var_default = 5 # Default value added at run time" << std::endl;

    std::istringstream s(ss.str());
    in.LoadFromStream(s);

    // capture all std::cout
    std::stringstream cout_cap;
    std::streambuf *cout = std::cout.rdbuf(cout_cap.rdbuf());

    WHEN("We require a paramter that has been provided") {
      THEN("Nothing should happen") {
        REQUIRE_NOTHROW(in.CheckRequired("block1", "var1"));
        REQUIRE_NOTHROW(in.CheckRequired("block2", "var4"));
        REQUIRE_NOTHROW(in.CheckRequired("block1", "var2"));
      }
    }
    AND_WHEN("We require missing parameters") {
      THEN("The check should throw a runtime error") {
        REQUIRE_THROWS_AS(in.CheckRequired("block2", "var9"), std::runtime_error);
      }
    }
    AND_WHEN("We require a parameter that is set by a code default") {
      THEN("The check should throw a runtime error") {
        REQUIRE_THROWS_AS(in.CheckRequired("block2", "var_default"), std::runtime_error);
      }
    }
    AND_WHEN("We desire missing parameters") {
      cout_cap.clear();
      THEN("The check should print warnings") {
        in.CheckDesired("block2", "var2");
        in.CheckDesired("block3", "var4");
        std::stringstream ss;
        ss << std::endl
           << "### WARNING in CheckDesired:" << std::endl
           << "Parameter file missing desired field <block2>/var2" << std::endl
           << std::endl
           << "### WARNING in CheckDesired:" << std::endl
           << "Parameter file missing desired field <block3>/var4" << std::endl;
        REQUIRE(cout_cap.str() == ss.str());
      }
    }
    std::cout.rdbuf(cout);
  }
  GIVEN("An invalid input deck") {
    ParameterInput in;
    std::stringstream ss;
    ss << "<block1>" << std::endl
       << "var1 = 0   # comment" << std::endl
       << "var2 = 1,  & 2.5 # another comment" << std::endl
       << "       2" << std::endl
       << "<block2>" << std::endl
       << "var3 = 3" << std::endl
       << "# comment" << std::endl
       << "var4 = 4" << std::endl
       << "var_default = 5 # Default value added at run time" << std::endl;
    WHEN("it is parsed") {
      std::istringstream s(ss.str());
      REQUIRE_THROWS_AS(in.LoadFromStream(s), std::runtime_error);
    }
  }

  GIVEN("An input deck with hidden characters and weird whitespace") {
    ParameterInput in;
    std::stringstream ss;
    ss << "<block1>" << std::endl
       << "  var1 = 0   # comment\r" << std::endl
       << "\tvar2 = 1,  \t& # another comment\n\r" << std::endl
       << "\t\t       2" << std::endl
       << "<block2>" << std::endl
       << " var3 = myval\r" << std::endl;

    std::istringstream s(ss.str());
    in.LoadFromStream(s);

    WHEN("We read the parameters") {
      THEN("They should be read correctly") {
        REQUIRE(in.GetInteger("block1", "var1") == 0);
        auto var2 = in.GetVector<int>("block1", "var2");
        REQUIRE(var2.size() == 2);
        if (var2.size() == 2) { // to guard against a segfault
          REQUIRE(((var2[0] == 1) && (var2[1] == 2)));
        }
        REQUIRE(in.GetString("block2", "var3") == "myval");
      }
    }
  }
}

TEST_CASE("Parameter inputs can be hashed and hashing provides useful sanity checks",
          "[ParameterInput][Hash]") {
  GIVEN("Two ParameterInput objects already populated") {
    ParameterInput in1, in2;
    std::hash<ParameterInput> hasher;
    std::stringstream ss;
    ss << "<block1>" << std::endl
       << "var1 = 0   # comment" << std::endl
       << "var2 = 1,  & # another comment" << std::endl
       << "       2" << std::endl
       << "<block2>" << std::endl
       << "var3 = 3" << std::endl
       << "# comment" << std::endl
       << "var4 = 4" << std::endl;

    // JMM: streams are stateful. Need to be very careful here.
    std::string ideck = ss.str();
    std::istringstream s1(ideck);
    std::istringstream s2(ideck);
    in1.LoadFromStream(s1);
    in2.LoadFromStream(s2);

    WHEN("We hash these parameter inputs") {
      std::size_t hash1 = hasher(in1);
      std::size_t hash2 = hasher(in2);
      THEN("The hashes agree") { REQUIRE(hash1 == hash2); }

      AND_WHEN("We modify both parameter inputs in the same way") {
        in1.GetOrAddReal("block3", "var5", 2.0);
        in2.GetOrAdd<parthenon::Real>("block3", "var5", 2.0);
        THEN("The hashes agree") {
          std::size_t hash1 = hasher(in1);
          std::size_t hash2 = hasher(in2);
          REQUIRE(hash1 == hash2);

          AND_WHEN("When we modify one input but not the other") {
            in2.GetOrAddInteger("block3", "var6", 7);
            THEN("The hashes will not agree") {
              std::size_t hash1 = hasher(in1);
              std::size_t hash2 = hasher(in2);
              REQUIRE(hash1 != hash2);
            }
          }
        }
      }
    }
  }
}

TEST_CASE("Test deleting parameters from ParameterInput", "[ParameterInput]") {
  GIVEN("A ParameterInput object already populated") {
    ParameterInput in;
    std::stringstream ss;
    ss << "<block1>" << std::endl
       << "var1 = 0   # comment" << std::endl
       << "var2 = 0   # comment" << std::endl
       << "<block2>" << std::endl
       << "var2 = 2" << std::endl;

    std::istringstream s(ss.str());
    in.LoadFromStream(s);

    THEN("block1/var1 exists") { REQUIRE(in.DoesParameterExist("block1", "var1")); }

    WHEN("We delete a parameter") {
      in.RemoveParameter("block1", "var1");
      THEN("It no longer exists") { REQUIRE(!in.DoesParameterExist("block1", "var1")); }
      THEN("And others still do") { REQUIRE(in.DoesParameterExist("block1", "var2")); }
    }
  }
}

// Phase 1 Tests: Map Resolution and Block Prefix Queries
TEST_CASE("ResolveParametersToMap populates internal map correctly",
          "[ParameterInput][Phase1]") {
  GIVEN("A ParameterInput with multiple blocks and parameters") {
    ParameterInput in;
    std::stringstream ss;
    ss << "<block1>" << std::endl
       << "int_param = 42" << std::endl
       << "real_param = 3.14" << std::endl
       << "bool_param = true" << std::endl
       << "string_param = hello" << std::endl
       << "<block2>" << std::endl
       << "vector_param = 1, 2, 3, 4" << std::endl;

    std::istringstream s(ss.str());
    in.LoadFromStream(s);

    WHEN("ResolveParametersToMap is called") {
      in.ResolveParametersToMap();

      THEN("All parameters remain accessible via Get methods") {
        REQUIRE(in.GetInteger("block1", "int_param") == 42);
        REQUIRE(in.GetReal("block1", "real_param") == Approx(3.14));
        REQUIRE(in.GetBoolean("block1", "bool_param") == true);
        REQUIRE(in.GetString("block1", "string_param") == "hello");

        auto vec = in.GetVector<int>("block2", "vector_param");
        REQUIRE(vec.size() == 4);
        REQUIRE(vec[0] == 1);
        REQUIRE(vec[3] == 4);
      }
    }
  }
}

TEST_CASE("GetBlocksWithPrefix returns matching blocks only",
          "[ParameterInput][Phase1]") {
  GIVEN("A ParameterInput with blocks having different prefixes") {
    ParameterInput in;
    std::stringstream ss;
    ss << "<parthenon/output1>" << std::endl
       << "dt = 0.1" << std::endl
       << "<parthenon/output2>" << std::endl
       << "dt = 0.2" << std::endl
       << "<parthenon/mesh>" << std::endl
       << "nx1 = 64" << std::endl
       << "<other/block>" << std::endl
       << "value = 5" << std::endl;

    std::istringstream s(ss.str());
    in.LoadFromStream(s);

    WHEN("GetBlocksWithPrefix is called for 'parthenon/output'") {
      auto blocks = in.GetBlocksWithPrefix("parthenon/output");

      THEN("It returns only the output blocks") {
        REQUIRE(blocks.size() == 2);
        REQUIRE(std::find(blocks.begin(), blocks.end(), "parthenon/output1") !=
                blocks.end());
        REQUIRE(std::find(blocks.begin(), blocks.end(), "parthenon/output2") !=
                blocks.end());
      }
    }
  }
}

TEST_CASE("Phase 1 type safety: wrong type access behavior", "[ParameterInput][Phase1]") {
  GIVEN("A ParameterInput with typed parameters") {
    ParameterInput in;
    std::stringstream ss;
    ss << "<types>" << std::endl
       << "int_val = 42" << std::endl
       << "string_val = hello" << std::endl
       << "bool_val = true" << std::endl;

    std::istringstream s(ss.str());
    in.LoadFromStream(s);
    in.ResolveParametersToMap();

    WHEN("We try to read a non-numeric string as a number") {
      THEN("GetInteger should fail during conversion") {
        REQUIRE_THROWS(in.GetInteger("types", "string_val"));
      }
      THEN("GetReal returns 0.0 for invalid strings (atof behavior)") {
        // Note: atof() doesn't throw, it just returns 0.0 for invalid input
        // This is existing behavior, not a Phase 1 bug
        REQUIRE(in.GetReal("types", "string_val") == Approx(0.0));
      }
    }

    WHEN("We try to read a boolean string as a number") {
      THEN("It should fail during conversion") {
        REQUIRE_THROWS(in.GetInteger("types", "bool_val"));
      }
    }
  }
}
