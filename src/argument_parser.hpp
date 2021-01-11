//========================================================================================
// (C) (or copyright) 2021. Triad National Security, LLC. All rights reserved.
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

#ifndef ARGUMENT_PARSER_HPP_
#define ARGUMENT_PARSER_HPP_

#include <iostream>
#include <string>

#include "defs.hpp"
#include "globals.hpp"
#include "utils/utils.hpp"

namespace parthenon {

enum class ArgStatus { ok, complete, error };

class ArgParse {
 public:
  ArgParse() = default;
  ArgStatus parse(int argc, char *argv[]) {
    for (int i = 1; i < argc; i++) {
      // If argv[i] is a 2 character string of the form "-?" then:
      if (*argv[i] == '-' && *(argv[i] + 1) != '\0' && *(argv[i] + 2) == '\0') {
        // check validity of command line options + arguments:
        char opt_letter = *(argv[i] + 1);
        switch (opt_letter) {
        // options that do not take arguments:
        case 'n':
        case 'c':
        case 'h':
          break;
          // options that require arguments:
        default:
          if ((i + 1 >= argc) // flag is at the end of the command line options
              || (*argv[i + 1] == '-')) { // flag is followed by another flag
            if (Globals::my_rank == 0) {
              std::cout << "### FATAL ERROR in main" << std::endl
                        << "-" << opt_letter << " must be followed by a valid argument\n";
              return ArgStatus::error;
            }
          }
        }
        switch (*(argv[i] + 1)) {
        case 'i': // -i <input_filename>
          input_filename = argv[++i];
          iarg_flag = 1;
          break;
        case 'r': // -r <restart_file>
          res_flag = 1;
          restart_filename = argv[++i];
          break;
        case 'd': // -d <run_directory>
          prundir = argv[++i];
          break;
        case 'n':
          narg_flag = 1;
          break;
        case 'm': // -m <nproc>
          mesh_flag = static_cast<int>(std::strtol(argv[++i], nullptr, 10));
          break;
        case 't': // -t <hh:mm:ss>
          int wth, wtm, wts;
          std::sscanf(argv[++i], "%d:%d:%d", &wth, &wtm, &wts);
          wtlim = wth * 3600 + wtm * 60 + wts;
          break;
        case 'c':
          if (Globals::my_rank == 0) ShowConfig();
          return ArgStatus::error;
        case 'h':
        default:
          if (Globals::my_rank == 0) {
            std::string athena_version = "version 19.0 - August 2019";
            std::cout << "Athena++ " << athena_version << std::endl;
            std::cout << "Usage: " << argv[0] << " [options] [block/par=value ...]\n";
            std::cout << "Options:" << std::endl;
            std::cout << "  -i <file>       specify input file [athinput]\n";
            std::cout << "  -r <file>       restart with this file\n";
            std::cout << "  -d <directory>  specify run dir [current dir]\n";
            std::cout << "  -n              parse input file and quit\n";
            std::cout << "  -c              show configuration and quit\n";
            std::cout << "  -m <nproc>      output mesh structure and quit\n";
            std::cout << "  -t hh:mm:ss     wall time limit for final output\n";
            std::cout << "  -h              this help\n";
            ShowConfig();
          }
          return ArgStatus::error;
        }
      } // else if argv[i] not of form "-?" ignore it here (tested in ModifyFromCmdline)
    }

    if (restart_filename == nullptr && input_filename == nullptr) {
      // no input file is given
      std::cout << "### FATAL ERROR in main" << std::endl
                << "No input file or restart file is specified." << std::endl;
      return ArgStatus::error;
    }
    return ArgStatus::ok;
  }

  char *input_filename = nullptr;
  char *restart_filename = nullptr;
  char *prundir = nullptr;
  int res_flag = 0;
  int narg_flag = 0;
  int iarg_flag = 0;
  int mesh_flag = 0;
  int wtlim = 0;
  int exit_flag = 0;
};

} // namespace parthenon

#endif // ARGUMENT_PARSER_HPP_
