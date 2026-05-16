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

#include <cstdio>
#include <istream>
#include <sstream>
#include <string>
#include <vector>

#include <catch2/catch.hpp>

#include "parameter_input.hpp"
#include "parameter_parsers/rummy_parser.hpp"

using parthenon::ParameterInput;

TEST_CASE("LoadFromRummyStream: basic scalar types", "[Rummy]") {
  GIVEN("A Rummy-format stream with bool, string, and numeric cards") {
    ParameterInput in;
    std::istringstream ss("<mesh>\n"
                          "nx = 64\n"
                          "cfl = 0.4\n"
                          "active = true\n"
                          "label = \"hydro\"\n");
    parthenon::LoadParameterFromRummy(in, ss, false);

    THEN("Integer parameter is readable") { REQUIRE(in.GetInteger("mesh", "nx") == 64); }
    THEN("Real parameter is readable") {
      REQUIRE(in.GetReal("mesh", "cfl") == Approx(0.4));
    }
    THEN("Boolean parameter is readable") {
      REQUIRE(in.GetBoolean("mesh", "active") == true);
    }
    THEN("String parameter is readable") {
      REQUIRE(in.GetString("mesh", "label") == "hydro");
    }
    THEN("Block exists") { REQUIRE(in.DoesBlockExist("mesh")); }
    THEN("Parameters exist") {
      REQUIRE(in.DoesParameterExist("mesh", "nx"));
      REQUIRE(in.DoesParameterExist("mesh", "cfl"));
    }
  }
}

TEST_CASE("LoadFromRummyStream: global variables go to '/' block", "[Rummy]") {
  GIVEN("A Rummy-format stream with global variables") {
    ParameterInput in;
    std::istringstream ss("Lx = 1.0\n"
                          "flag = false\n"
                          "name = \"global_scope\"\n"
                          "<mesh>\n"
                          "nx = 10\n");
    parthenon::LoadParameterFromRummy(in, ss, false);

    THEN("Globals are stored under the '/' block") {
      REQUIRE(in.DoesParameterExist("/", "Lx"));
      REQUIRE(in.GetReal("/", "Lx") == Approx(1.0));
      REQUIRE(in.DoesParameterExist("/", "flag"));
      REQUIRE(in.GetBoolean("/", "flag") == false);
      REQUIRE(in.GetString("/", "name") == "global_scope");
    }
    THEN("Non-global parameters are unaffected") {
      REQUIRE(in.GetInteger("mesh", "nx") == 10);
    }
  }
}

TEST_CASE("LoadFromRummyStream: numeric vector reconstruction", "[Rummy]") {
  GIVEN("A Rummy stream with a vector of reals and a vector of ints") {
    ParameterInput in;

    std::istringstream ss("<block>\n"
                          "vals = [1.5, 2.5, 3.5]\n"
                          "counts = [10, 20, 30]\n");
    parthenon::LoadParameterFromRummy(in, ss, false);

    THEN("Real vector is reconstructed correctly") {
      auto v = in.GetVector<parthenon::Real>("block", "vals");
      REQUIRE(v.size() == 3);
      REQUIRE(v[0] == Approx(1.5));
      REQUIRE(v[1] == Approx(2.5));
      REQUIRE(v[2] == Approx(3.5));
    }
    THEN("Integer vector is reconstructed correctly") {
      auto v = in.GetVector<int>("block", "counts");
      REQUIRE(v.size() == 3);
      REQUIRE(v[0] == 10);
      REQUIRE(v[1] == 20);
      REQUIRE(v[2] == 30);
    }
  }
}

TEST_CASE("LoadFromRummyStream: string vector reconstruction", "[Rummy]") {
  GIVEN("A Rummy stream with a vector of strings") {
    ParameterInput in;
    std::istringstream ss("<block>\n"
                          "tags = [\"alpha\", \"beta\", \"gamma\"]\n");
    parthenon::LoadParameterFromRummy(in, ss, false);

    THEN("String vector is reconstructed correctly") {
      auto v = in.GetVector<std::string>("block", "tags");
      REQUIRE(v.size() == 3);
      REQUIRE(v[0] == "alpha");
      REQUIRE(v[1] == "beta");
      REQUIRE(v[2] == "gamma");
    }
  }
}

TEST_CASE("LoadFromRummyStream: expressions are evaluated", "[Rummy]") {
  GIVEN("A Rummy stream with arithmetic expressions and cross-suit references") {
    ParameterInput in;
    std::istringstream ss("base = 4.0\n"
                          "<block>\n"
                          "doubled = base * 2.0\n"
                          "squared = base**2\n");
    parthenon::LoadParameterFromRummy(in, ss, false);

    THEN("Expressions are fully evaluated before storage") {
      REQUIRE(in.GetReal("block", "doubled") == Approx(8.0));
      REQUIRE(in.GetReal("block", "squared") == Approx(16.0));
    }
  }
}

TEST_CASE("IsRummyFormat: detects Rummy vs legacy format", "[Rummy]") {
  GIVEN("A legacy-format input file (block header before any value)") {
    std::istringstream ss("<mesh>\n"
                          "nx1 = 64\n"
                          "nx2 = 32\n");
    THEN("IsRummyFormat returns false") {
      REQUIRE(parthenon::IsRummyFormat(ss, false) == false);
    }
  }

  GIVEN("A Rummy-format file: global variable before first block") {
    std::istringstream ss("Lx = 1.0\n"
                          "<mesh>\n"
                          "nx = 64\n");
    THEN("IsRummyFormat returns true") {
      REQUIRE(parthenon::IsRummyFormat(ss, false) == true);
    }
  }

  GIVEN("A Rummy-format file: relative suit path <../") {
    std::istringstream ss("<physics>\n"
                          "hydro = true\n"
                          "<../eos>\n"
                          "gamma = 1.4\n");
    THEN("IsRummyFormat returns true") {
      REQUIRE(parthenon::IsRummyFormat(ss, false) == true);
    }
  }

  GIVEN("A Rummy-format file: ** power operator in a value") {
    std::istringstream ss("<block>\n"
                          "val = 2**10\n");
    THEN("IsRummyFormat returns true") {
      REQUIRE(parthenon::IsRummyFormat(ss, false) == true);
    }
  }

  GIVEN("A Rummy-format file: first line is '# use rummy'") {
    std::istringstream ss("# Use Rummy\n"
                          "<mesh>\n"
                          "nx = 64\n");
    THEN("IsRummyFormat returns true") {
      REQUIRE(parthenon::IsRummyFormat(ss, false) == true);
    }
  }

  GIVEN("A Rummy-format file: quoted string value") {
    std::istringstream ss("<mesh>\n"
                          "label = \"hydro\"\n");
    THEN("IsRummyFormat returns true") {
      REQUIRE(parthenon::IsRummyFormat(ss, false) == true);
    }
  }

  GIVEN("A Rummy-format file: bracket vector syntax in a value") {
    std::istringstream ss("<mesh>\n"
                          "nx = [64, 32, 16]\n");
    THEN("IsRummyFormat returns true") {
      REQUIRE(parthenon::IsRummyFormat(ss, false) == true);
    }
  }

  GIVEN("A Rummy-format file: bracket slice syntax on the LHS") {
    std::istringstream ss("<mesh>\n"
                          "nx[:2] = [64, 32]\n");
    THEN("IsRummyFormat returns true") {
      REQUIRE(parthenon::IsRummyFormat(ss, false) == true);
    }
  }
}

TEST_CASE("LoadFromRummyStream: ModifyFromCmdline overrides Rummy params", "[Rummy]") {
  GIVEN("A Rummy stream with a parameter") {
    ParameterInput in;
    std::istringstream ss("<mesh>\nnx = 32\n");
    parthenon::LoadParameterFromRummy(in, ss, false);

    WHEN("ModifyFromCmdline overrides   the parameter") {
      std::istringstream ss2("mesh.nx = 128\n");
      parthenon::LoadParameterFromRummy(in, ss2, true);
      THEN("The override wins") { REQUIRE(in.GetInteger("mesh", "nx") == 128); }
    }
  }
}

TEST_CASE("LoadFromRummyStream: comma-separated vector without brackets", "[Rummy]") {
  GIVEN("A Rummy stream using bare comma-separated syntax") {
    ParameterInput in;
    std::istringstream ss("<block>\n"
                          "vals = 1.0, 2.0, 3.0\n"
                          "counts = 10, 20, 30\n");
    parthenon::LoadParameterFromRummy(in, ss, false);

    THEN("Real vector is reconstructed correctly") {
      auto v = in.GetVector<parthenon::Real>("block", "vals");
      REQUIRE(v.size() == 3);
      REQUIRE(v[0] == Approx(1.0));
      REQUIRE(v[1] == Approx(2.0));
      REQUIRE(v[2] == Approx(3.0));
    }
    THEN("Integer vector is reconstructed correctly") {
      auto v = in.GetVector<int>("block", "counts");
      REQUIRE(v.size() == 3);
      REQUIRE(v[0] == 10);
      REQUIRE(v[1] == 20);
      REQUIRE(v[2] == 30);
    }
  }
}

TEST_CASE("LoadFromRummyStream: slice assignment syntax", "[Rummy]") {
  GIVEN("A Rummy stream using slice assignment v[:N] = [...]") {
    ParameterInput in;
    std::istringstream ss("<block>\n"
                          "v[:3] = [100, 200, 300]\n");
    parthenon::LoadParameterFromRummy(in, ss, false);

    THEN("Vector is reconstructed correctly from slice assignment") {
      auto v = in.GetVector<int>("block", "v");
      REQUIRE(v.size() == 3);
      REQUIRE(v[0] == 100);
      REQUIRE(v[1] == 200);
      REQUIRE(v[2] == 300);
    }
  }
}

TEST_CASE("LoadFromRummyStream: cross-block references are evaluated", "[Rummy]") {
  GIVEN("A Rummy stream where one block references another block's variable") {
    ParameterInput in;
    std::istringstream ss("<physics>\n"
                          "gamma = 1.4\n"
                          "<eos>\n"
                          "gamma_minus_one = physics.gamma - 1.0\n"
                          "gamma_sq = physics.gamma ** 2\n");
    LoadParameterFromRummy(in, ss, false);

    THEN("Cross-block reference is fully evaluated before storage") {
      REQUIRE(in.GetReal("eos", "gamma_minus_one") == Approx(0.4));
      REQUIRE(in.GetReal("eos", "gamma_sq") == Approx(1.96));
    }
  }
}

TEST_CASE("LoadFromRummyStream: global variables accessible from blocks", "[Rummy]") {
  GIVEN("A Rummy stream with a global variable used inside a block") {
    ParameterInput in;
    std::istringstream ss("Lx = 10.0\n"
                          "<mesh>\n"
                          "dx = Lx / 100\n"
                          "half_Lx = Lx * 0.5\n");
    parthenon::LoadParameterFromRummy(in, ss, false);

    THEN("Global is stored under the '/' block") {
      REQUIRE(in.GetReal("/", "Lx") == Approx(10.0));
    }
    THEN("Block parameters referencing the global are evaluated") {
      REQUIRE(in.GetReal("mesh", "dx") == Approx(0.1));
      REQUIRE(in.GetReal("mesh", "half_Lx") == Approx(5.0));
    }
  }
}

static std::string captureStdout(std::function<void()> f) {
  int pipefd[2];
  pipe(pipefd);
  int saved = dup(STDOUT_FILENO);
  dup2(pipefd[1], STDOUT_FILENO);
  close(pipefd[1]);

  f();
  fflush(stdout);

  dup2(saved, STDOUT_FILENO);
  close(saved);

  std::string result;
  char buf[256];
  ssize_t n;
  while ((n = read(pipefd[0], buf, sizeof(buf))) > 0)
    result.append(buf, n);
  close(pipefd[0]);
  return result;
}

TEST_CASE("LoadFromRummyStream: print statement outside a block", "[Rummy]") {
  GIVEN("A Rummy stream with a print statement before any block") {
    ParameterInput in;
    // print is a Rummy/pips statement; it produces output but no card.
    // Verify it doesn't crash and doesn't appear as a parameter.

    // capture stdout to verify print statement doesn't produce stored parameter but does
    // produce output

    std::istringstream ss("x = 42.0\n"
                          "print(x)\n"
                          "<block>\n"
                          "y = x + 1\n");

    THEN("LoadFromRummyStream completes without error") {
      REQUIRE_NOTHROW(LoadParameterFromRummy(in, ss, false));
    }
    AND_THEN("The print statement produces no stored parameter") {
      std::istringstream ss2("x = 42.0\n"
                             "print(x)\n"
                             "<block>\n"
                             "y = x + 1\n");

      std::string dummy_cout =
          captureStdout([&]() { parthenon::LoadParameterFromRummy(in, ss2, false); });
      REQUIRE_FALSE(in.DoesParameterExist("/", "print"));
      REQUIRE(in.GetReal("block", "y") == Approx(43.0));
      REQUIRE(dummy_cout.substr(0, 2) == "42");
    }
  }
}

TEST_CASE("LoadFromRummyStream: vector slice with element-wise math", "[Rummy]") {
  GIVEN("A Rummy stream that defines a 3-element vector, then cubes a 2-element "
        "sub-slice") {
    ParameterInput in;
    // base[:3] defines [2.0, 3.0, 4.0].
    // cubed[:2] = base[:2] ** 3 takes only the first two elements and cubes them.
    std::istringstream ss("<block>\n"
                          "base[:3] = [2.0, 3.0, 4.0]\n"
                          "cubed[:2] = base[:2] ** 3\n");
    parthenon::LoadParameterFromRummy(in, ss, false);

    THEN("Base vector retains all three elements") {
      auto b = in.GetVector<parthenon::Real>("block", "base");
      REQUIRE(b.size() == 3);
      REQUIRE(b[0] == Approx(2.0));
      REQUIRE(b[1] == Approx(3.0));
      REQUIRE(b[2] == Approx(4.0));
    }
    THEN("Cubed slice contains only the first two elements, each cubed") {
      auto c = in.GetVector<parthenon::Real>("block", "cubed");
      REQUIRE(c.size() == 2);
      REQUIRE(c[0] == Approx(8.0));  // 2^3
      REQUIRE(c[1] == Approx(27.0)); // 3^3
    }
  }
}

TEST_CASE("LoadFromRummyStream: second stream overwrites existing parameters",
          "[Rummy]") {
  GIVEN("A first Rummy stream establishing initial values") {
    ParameterInput in;
    std::istringstream ss1("<mesh>\n"
                           "nx = 64\n"
                           "cfl = 0.3\n"
                           "<physics>\n"
                           "gamma = 1.4\n");
    parthenon::LoadParameterFromRummy(in, ss1, false);

    WHEN("A second Rummy stream updates some of those parameters") {
      std::istringstream ss2("<mesh>\n"
                             "nx = 128\n"
                             "cfl = 0.5\n");
      parthenon::LoadParameterFromRummy(in, ss2, true);

      THEN("Updated parameters reflect the second stream") {
        REQUIRE(in.GetInteger("mesh", "nx") == 128);
        REQUIRE(in.GetReal("mesh", "cfl") == Approx(0.5));
      }
      THEN("Parameters not present in the second stream are unchanged") {
        REQUIRE(in.GetReal("physics", "gamma") == Approx(1.4));
      }
    }
  }
}
