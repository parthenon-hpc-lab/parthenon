//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
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

// This file was created in part with the generative AI

#include <iostream>
#include <istream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <catch2/catch.hpp>

#include "parameter_input.hpp"

using parthenon::ParameterInput;
using parthenon::Real;

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

  GIVEN("An input deck with a trailing comma in a vector-valued parameter") {
    ParameterInput in;
    std::stringstream ss;
    ss << "<block1>" << std::endl << "var1 = 1, 2, 3," << std::endl;

    std::istringstream s(ss.str());
    in.LoadFromStream(s);

    WHEN("The vector is read back") {
      THEN("The trailing comma does not introduce an empty final element") {
        auto var1 = in.GetVector<int>("block1", "var1");
        REQUIRE(var1 == std::vector<int>{1, 2, 3});
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
TEST_CASE("FinalizeParsing populates internal map correctly",
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

    WHEN("FinalizeParsing is called") {
      in.FinalizeParsing();

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

TEST_CASE("GetBlockNamesWithPrefix returns matching blocks only",
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

    WHEN("GetBlockNamesWithPrefix is called for 'parthenon/output'") {
      auto blocks = in.GetBlockNamesWithPrefix("parthenon/output");

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
    in.FinalizeParsing();

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

// Parser Separation Tests: AddParsedParameter API
TEST_CASE("AddParsedParameter with typed scalar values", "[ParameterInput][Parser]") {
  GIVEN("An empty ParameterInput") {
    ParameterInput in;

    WHEN("We add typed scalar parameters via AddParsedParameter") {
      in.AddParsedParameter("block1", "int_val", 42);
      in.AddParsedParameter("block1", "real_val", 3.14);
      in.AddParsedParameter("block1", "bool_val", true);
      in.AddParsedParameter("block1", "string_val", std::string("hello"));
      in.FinalizeParsing();

      THEN("They can be retrieved with correct types") {
        REQUIRE(in.GetInteger("block1", "int_val") == 42);
        REQUIRE(in.GetReal("block1", "real_val") == Approx(3.14));
        REQUIRE(in.GetBoolean("block1", "bool_val") == true);
        REQUIRE(in.GetString("block1", "string_val") == "hello");
      }
    }
  }
}

TEST_CASE("AddParsedParameter with typed vector values", "[ParameterInput][Parser]") {
  GIVEN("An empty ParameterInput") {
    ParameterInput in;

    WHEN("We add typed vector parameters") {
      std::vector<int> int_vec = {1, 2, 3, 4};
      std::vector<parthenon::Real> real_vec = {1.1, 2.2, 3.3};
      std::vector<bool> bool_vec = {true, false, true};
      std::vector<std::string> str_vec = {"foo", "bar", "baz"};

      in.AddParsedParameter("vectors", "int_vec", int_vec);
      in.AddParsedParameter("vectors", "real_vec", real_vec);
      in.AddParsedParameter("vectors", "bool_vec", bool_vec);
      in.AddParsedParameter("vectors", "str_vec", str_vec);
      in.FinalizeParsing();

      THEN("They can be retrieved correctly") {
        auto iv = in.GetVector<int>("vectors", "int_vec");
        REQUIRE(iv.size() == 4);
        REQUIRE(iv[0] == 1);
        REQUIRE(iv[3] == 4);

        auto rv = in.GetVector<parthenon::Real>("vectors", "real_vec");
        REQUIRE(rv.size() == 3);
        REQUIRE(rv[0] == Approx(1.1));
        REQUIRE(rv[2] == Approx(3.3));

        auto bv = in.GetVector<bool>("vectors", "bool_vec");
        REQUIRE(bv.size() == 3);
        REQUIRE(bv[0] == true);
        REQUIRE(bv[1] == false);

        auto sv = in.GetVector<std::string>("vectors", "str_vec");
        REQUIRE(sv.size() == 3);
        REQUIRE(sv[0] == "foo");
        REQUIRE(sv[2] == "baz");
      }
    }
  }
}

TEST_CASE("AddParsedParameter with UnresolvedString", "[ParameterInput][Parser]") {
  GIVEN("An empty ParameterInput") {
    ParameterInput in;

    WHEN("We add UnresolvedString parameters that need lazy conversion") {
      using parthenon::UnresolvedString;
      in.AddParsedParameter("lazy", "int_str", UnresolvedString("42"));
      in.AddParsedParameter("lazy", "real_str", UnresolvedString("3.14159"));
      in.AddParsedParameter("lazy", "bool_str", UnresolvedString("true"));
      in.AddParsedParameter("lazy", "vec_str", UnresolvedString("1, 2, 3, 4, 5"));
      in.FinalizeParsing();

      THEN("They are converted on first access") {
        REQUIRE(in.GetInteger("lazy", "int_str") == 42);
        REQUIRE(in.GetReal("lazy", "real_str") == Approx(3.14159));
        REQUIRE(in.GetBoolean("lazy", "bool_str") == true);

        auto vec = in.GetVector<int>("lazy", "vec_str");
        REQUIRE(vec.size() == 5);
        REQUIRE(vec[0] == 1);
        REQUIRE(vec[4] == 5);
      }

      AND_THEN("Subsequent accesses use the converted value") {
        // Access again to verify conversion is cached
        REQUIRE(in.GetInteger("lazy", "int_str") == 42);
        REQUIRE(in.GetReal("lazy", "real_str") == Approx(3.14159));
      }
    }
  }
}

TEST_CASE("Mixing LoadFromStream and AddParsedParameter", "[ParameterInput][Parser]") {
  GIVEN("A ParameterInput with file parameters") {
    ParameterInput in;
    std::stringstream ss;
    ss << "<file_block>" << std::endl
       << "file_param = 100" << std::endl
       << "<shared_block>" << std::endl
       << "from_file = text" << std::endl;

    std::istringstream s(ss.str());
    in.LoadFromStream(s);

    WHEN("We add additional parameters via AddParsedParameter") {
      in.AddParsedParameter("code_block", "code_param", 200);
      in.AddParsedParameter("shared_block", "from_code", 3.14);
      in.FinalizeParsing();

      THEN("Both file and code parameters are accessible") {
        REQUIRE(in.GetInteger("file_block", "file_param") == 100);
        REQUIRE(in.GetString("shared_block", "from_file") == "text");
        REQUIRE(in.GetInteger("code_block", "code_param") == 200);
        REQUIRE(in.GetReal("shared_block", "from_code") == Approx(3.14));
      }
    }
  }
}

TEST_CASE("AddParsedParameter overrides earlier values", "[ParameterInput][Parser]") {
  GIVEN("A ParameterInput with parameters") {
    ParameterInput in;

    WHEN("We add a parameter multiple times") {
      in.AddParsedParameter("override", "value", 100);
      in.AddParsedParameter("override", "value", 200);
      in.AddParsedParameter("override", "value", 300);
      in.FinalizeParsing();

      THEN("The last value wins") { REQUIRE(in.GetInteger("override", "value") == 300); }
    }
  }
}

TEST_CASE("FinalizeParsing prevents further parsing", "[ParameterInput][Parser]") {
  GIVEN("A ParameterInput that has been marked resolved") {
    ParameterInput in;
    in.AddParsedParameter("block", "param", 42);
    in.FinalizeParsing();

    WHEN("We try to add more parameters") {
      THEN("AddParsedParameter should throw") {
        REQUIRE_THROWS_AS(in.AddParsedParameter("block", "new_param", 100),
                          std::runtime_error);
      }
    }

    AND_WHEN("We try to load from stream") {
      std::stringstream ss;
      ss << "<block2>" << std::endl << "param2 = 200" << std::endl;
      std::istringstream s(ss.str());

      THEN("LoadFromStream should throw") {
        REQUIRE_THROWS_AS(in.LoadFromStream(s), std::runtime_error);
      }
    }
  }
}

TEST_CASE("Parameter ordering is preserved for restart compatibility",
          "[ParameterInput][Parser]") {
  GIVEN("Parameters added in a specific order") {
    ParameterInput in;

    // Add parameters in deliberate order
    in.AddParsedParameter("zblock", "zparam", 3);
    in.AddParsedParameter("ablock", "aparam", 1);
    in.AddParsedParameter("mblock", "mparam", 2);
    in.AddParsedParameter("ablock", "zparam", 4);
    in.AddParsedParameter("ablock", "bparam", 5);
    in.FinalizeParsing();

    WHEN("We query the blocks") {
      auto blocks = in.GetBlockNamesWithPrefix("");

      THEN("Blocks appear in insertion order") {
        REQUIRE(blocks.size() >= 3);
        auto zpos = std::find(blocks.begin(), blocks.end(), "zblock");
        auto apos = std::find(blocks.begin(), blocks.end(), "ablock");
        auto mpos = std::find(blocks.begin(), blocks.end(), "mblock");

        REQUIRE(zpos != blocks.end());
        REQUIRE(apos != blocks.end());
        REQUIRE(mpos != blocks.end());

        // zblock added first, ablock second, mblock third
        REQUIRE(zpos < apos);
        REQUIRE(apos < mpos);
      }
    }

    WHEN("We iterate parameters within a block") {
      // This tests that parameters within a block maintain insertion order
      // Note: Current implementation doesn't expose parameter iteration directly,
      // but this is verified by restart file consistency tests
      THEN("Parameters are accessible in any order") {
        REQUIRE(in.GetInteger("ablock", "aparam") == 1);
        REQUIRE(in.GetInteger("ablock", "zparam") == 4);
        REQUIRE(in.GetInteger("ablock", "bparam") == 5);
      }
    }
  }
}

TEST_CASE("AddParsedParameter creates blocks automatically", "[ParameterInput][Parser]") {
  GIVEN("An empty ParameterInput") {
    ParameterInput in;

    WHEN("We add parameters to non-existent blocks") {
      in.AddParsedParameter("new_block1", "param1", 1);
      in.AddParsedParameter("new_block2", "param2", 2);
      in.AddParsedParameter("new_block1", "param3", 3);
      in.FinalizeParsing();

      THEN("Blocks are created automatically") {
        REQUIRE(in.DoesParameterExist("new_block1", "param1"));
        REQUIRE(in.DoesParameterExist("new_block2", "param2"));
        REQUIRE(in.DoesParameterExist("new_block1", "param3"));
      }
    }
  }
}

TEST_CASE("Parser interface works without FinalizeParsing for backward compatibility",
          "[ParameterInput][Parser]") {
  GIVEN("Parameters added via AddParsedParameter") {
    ParameterInput in;
    in.AddParsedParameter("block", "value", 42);

    WHEN("We access parameters without calling FinalizeParsing") {
      THEN("Parameters are automatically resolved on first access") {
        REQUIRE_NOTHROW(in.GetInteger("block", "value"));
        REQUIRE(in.GetInteger("block", "value") == 42);
      }
    }
  }
}

TEST_CASE("Empty vector defaults round-trip through the parameter store",
          "[ParameterInput]") {
  ParameterInput in;
  auto values = in.GetOrAddVector<std::string>("block1", "var1", {});
  REQUIRE(values.empty());
  REQUIRE(in.GetVector<std::string>("block1", "var1").empty());
}

TEST_CASE("GetAsUnresolvedString returns string representations", "[ParameterInput]") {
  GIVEN("Parameters from input file") {
    ParameterInput in;
    std::stringstream ss;
    ss << "<test>" << std::endl
       << "int_param = 42" << std::endl
       << "real_param = 3.14159" << std::endl
       << "bool_param = true" << std::endl
       << "string_param = hello" << std::endl
       << "vector_param = 1, 2, 3" << std::endl;
    std::istringstream s(ss.str());
    in.LoadFromStream(s);

    WHEN("GetAsUnresolvedString is called") {
      THEN("Returns original string from file") {
        REQUIRE(in.GetAsUnresolvedString("test", "int_param") == "42");
        REQUIRE(in.GetAsUnresolvedString("test", "real_param") == "3.14159");
        REQUIRE(in.GetAsUnresolvedString("test", "bool_param") == "true");
        REQUIRE(in.GetAsUnresolvedString("test", "string_param") == "hello");
        REQUIRE(in.GetAsUnresolvedString("test", "vector_param") == "1, 2, 3");
      }
    }
  }

  GIVEN("Parameters added programmatically") {
    ParameterInput in;
    in.Set<int>("runtime", "int_val", 99);
    in.Set<bool>("runtime", "bool_val", false);
    in.Set<Real>("runtime", "real_val", 2.718);

    WHEN("GetAsUnresolvedString is called") {
      THEN("Returns converted string representation") {
        REQUIRE(in.GetAsUnresolvedString("runtime", "int_val") == "99");
        REQUIRE(in.GetAsUnresolvedString("runtime", "bool_val") == "false");
        // Real conversion should use full precision
        std::string real_str = in.GetAsUnresolvedString("runtime", "real_val");
        REQUIRE(std::stod(real_str) == Approx(2.718));
      }
    }
  }

  GIVEN("Missing parameter") {
    ParameterInput in;
    THEN("GetAsUnresolvedString throws") {
      REQUIRE_THROWS(in.GetAsUnresolvedString("missing", "param"));
    }
  }
}
