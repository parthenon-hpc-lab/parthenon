//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020-2024. Triad National Security, LLC. All rights reserved.
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
//! \file parameter_input.cpp
//  \brief implementation of functions in class ParameterInput
//
// PURPOSE: Member functions of this class are used to read and parse the input file.
//   Functionality is loosely modeled after FORTRAN namelist.
//
// EXAMPLE of input file in 'Athena++' format:
//   <blockname1>      # block name; must be on a line by itself
//                     # everything after a hash symbol is a comment and is ignored
//   name1=value       # each parameter name must be on a line by itself
//   name2 = value1    # whitespace around the = is optional
//                     # blank lines are OK
//   # my comment here   comment lines are OK
//   # name3 = value3    values (and blocks) that are commented out are ignored
//
//   <blockname2>      # start new block
//   name1 = value1    # note that same parameter names can appear in different blocks
//   name2 = value2    # empty lines (like following) are OK
//
//   <blockname1>      # same blockname can re-appear, although NOT recommended
//   name3 = value3    # this would be the 3rd parameter name in blockname1
//   name1 = value4    # if parameter name is repeated, previous value is overwritten!
//
// LIMITATIONS:
//   - parameter specification (name=val #comment) must all be on a single line
//
// HISTORY:
//   - Nov 2002:  Created for Athena1.0/Cambridge release by Peter Teuben
//   - 2003-2008: Many improvements and extensions by T. Gardiner and J.M. Stone
//   - Jan 2014:  Rewritten in C++ for the Athena++ code by J.M. Stone
//========================================================================================

#include "parameter_input.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <toml.hpp>

#include "globals.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {

toml::table ParameterInput::Blocks() { return parameters_; }
toml::table ParameterInput::Blocks(const char *path) {
  return GetPath<toml::table>(path);
}
toml::table ParameterInput::Blocks(std::string &path) {
  return GetPath<toml::table>(path);
}
// Get const copies of entire contents, primarily for hashing
const toml::table ParameterInput::GetAll() const { return parameters_; }
const toml::table ParameterInput::GetAllOrigins() const { return origins_; }

void ParameterInput::LoadFile(const std::string fname, bool check_for_overrides) {
  std::stringstream contents;
  char *buf = reinterpret_cast<char *>(calloc(sizeof(char), max_input_filesize_));

  IOWrapper infile;
  infile.Open(fname.c_str(), IOWrapper::FileMode::read);
  infile.Read_all(buf, sizeof(char), max_input_filesize_);
  infile.Close();
  contents.write(buf, max_input_filesize_);
  free(buf);

  LoadFromStream(contents, fname, check_for_overrides);

  return;
}

void ParameterInput::LoadFromStream(std::istream &is, std::string fname,
                                    bool check_for_overrides) {
  std::stringstream ss;
  ss << is.rdbuf();
  std::string input = ss.str();
  // Remove all null bytes: it's faster & toml doesn't like them
  input.erase(std::remove(input.begin(), input.end(), '\00'), input.end());
  // If n(<) > n([), we're parsing an old-style file. Otherwise, TOML
  int nangle = std::count(input.begin(), input.end(), '<');
  int nsquare = std::count(input.begin(), input.end(), '[');
  parameter_lists_.push_back(toml::table());
  auto &new_parameters = parameter_lists_.back();
  if (nangle > nsquare) {
    is.seekg(is.beg);
    new_parameters = LegacyParse(is, fname);
  } else {
    new_parameters = toml::parse(input);
  }
  // Merge from different inputs, only check for overrides if asked
  Merge(parameters_, new_parameters, check_for_overrides);

  // Now update origins
  if (fname == "") {
    SetOrigin(origins_, new_parameters, "restart");
  } else {
    SetOrigin(origins_, new_parameters, fname);
  }
}

void ParameterInput::ModifyFromCmdline(int argc, char *argv[]) {
  std::string input_text, path, value;
  std::stringstream msg;

  for (int i = 1; i < argc; i++) {
    input_text = argv[i];
    std::size_t equal_posn = input_text.find_first_of("="); // first "=" character

    // Only parse arguments with '='
    if (equal_posn == std::string::npos) continue;

    // Sanitize block + name together, but only if there are no dots
    path = input_text.substr(0, equal_posn);
    // This replaces all '/', so they can't be used in varnames which will be passed
    // on the command line, even when using pure TOML inputs.
    // ('/' obviously can't be used in block names anyway if we want back-compat)
    // Alternatively we could check for '.', which would forbid '.' in old-style
    // blocks and varnames retroactively.
    path = Path_(path, "");
    value = input_text.substr(equal_posn + 1, std::string::npos);

    if (!parameters_.at_path(path)) {
      if (Globals::my_rank == 0) {
        msg << "In function [ParameterInput::ModifyFromCmdline]:" << std::endl
            << "               Parameter '" << path
            << "' on command line not found in input/restart file. Parameter will be "
               "added.";
        PARTHENON_WARN(msg);
      }
    }
    // Commandline parameters can override anything or each other, don't check anything
    AddParameter_(parameters_, path, value, ParameterOrigin::cmdline);
  }
}

int ParameterInput::DoesParameterExist(const std::string &block,
                                       const std::string &name) {
  return DoesParameterExist(Path_(block, name));
}
int ParameterInput::DoesParameterExist(const std::string &path) {
  return !!parameters_.at_path(path);
}
int ParameterInput::DoesBlockExist(const std::string &block) {
  return parameters_.contains(Path_(block, ""));
}

ParameterOrigin ParameterInput::GetOrigin(const std::string &block,
                                          const std::string &name) {
  return GetOrigin(Path_(block, name));
}
ParameterOrigin ParameterInput::GetOrigin(const std::string &path) {
  if (!origins_.at_path(path)) {
    std::stringstream ss;
    ss << std::endl
       << "### ERROR in GetOrigin:" << std::endl
       << "Path " << path << " does not exist in origins_!" << std::endl
       << "This is a Parthenon issue" << std::endl;
    std::cerr << ss.str();
    return ParameterOrigin::restart;
    // throw std::runtime_error(ss.str());
  }
  std::string origin_name = origins_.at_path(path).ref<std::string>();
  // restart, input, cmdline, code
  if (origin_name == "restart") {
    return ParameterOrigin::restart;
  } else if (origin_name == "cmdline") {
    return ParameterOrigin::cmdline;
  } else if (origin_name == "code") {
    return ParameterOrigin::code;
  } else if (origin_name == "default") {
    return ParameterOrigin::defaultvalue;
  } else {
    // Otherwise this name reflects the input file
    return ParameterOrigin::input;
  }
  // Technically the user can set stuff directly, handle error with ::code?
}

std::string ParameterInput::GetOriginFile(const std::string &block,
                                          const std::string &name) {
  return GetOriginFile(Path_(block, name));
}
std::string ParameterInput::GetOriginFile(const std::string &path) {
  std::string origin_name = origins_.at_path(path).ref<std::string>();
  // restart, input, cmdline, code
  if (origin_name == "restart" || origin_name == "cmdline" || origin_name == "code" ||
      origin_name == "default") {
    // Throw
    return "none";
  } else {
    return origin_name;
  }
}

void ParameterInput::CheckRequired(const std::string &block, const std::string &name) {
  return CheckRequired(Path_(block, name));
}
void ParameterInput::CheckRequired(const std::string &path) {
  bool exists =
      DoesParameterExist(path) && (GetOrigin(path) != ParameterOrigin::defaultvalue);
  if (!exists) {
    std::stringstream ss;
    ss << std::endl
       << "### ERROR in CheckRequired:" << std::endl
       << "Parameter file missing required field " << path << std::endl
       << std::endl;
    throw std::runtime_error(ss.str());
  }
}

void ParameterInput::CheckDesired(const std::string &block, const std::string &name) {
  return CheckDesired(Path_(block, name));
}
void ParameterInput::CheckDesired(const std::string &path) {
  bool missing = true;
  bool defaulted = false;
  if (DoesParameterExist(path)) {
    missing = false;
    defaulted = (GetOrigin(path) == ParameterOrigin::defaultvalue);
  }
  if (missing) {
    std::cout << std::endl
              << "### WARNING in CheckDesired:" << std::endl
              << "Parameter file missing desired field " << path << std::endl;
  }
  if (defaulted) {
    std::cout << std::endl
              << "Defaulting to " << path << " = " << GetPath<std::string>(path)
              << std::endl;
  }
}

void ParameterInput::ParameterDump(std::ostream &os) { os << parameters_ << "\n"; }

toml::table ParameterInput::LegacyParse(std::istream &is, std::string fname) {
  std::string line, block_name, param_name, param_value, param_comment;
  std::size_t first_char, last_char;
  std::stringstream msg;
  int blocks_found{0};

  // Buffer multiple lines if a continuation character is present
  std::string multiline_name, multiline_value, multiline_comment;
  // Status in/out of continuation
  bool continuing = false;

  // Table for accumulating results
  toml::table tmp_tbl = toml::table();

  while (is.good()) {
    std::getline(is, line);

    // remove all \t\f\n\r\v but leave pure spaces
    line.erase(std::remove_if(line.begin(), line.end(),
                              [](char c) { return std::isspace(c) && c != ' '; }),
               line.end());

    if (line.empty()) continue;                               // skip blank line
    first_char = line.find_first_not_of(" ");                 // skip white space
    if (first_char == std::string::npos) continue;            // line is all white space
    if (line.compare(first_char, 1, "#") == 0) continue;      // skip comments
    if (line.compare(first_char, 9, "<par_end>") == 0) break; // stop on <par_end>

    if (line.compare(first_char, 1, "<") == 0) { // a new block
      if (continuing) {
        msg << "### FATAL ERROR in function [ParameterInput::LegacyParse]" << std::endl
            << "Multiline field ended unexpectedly with new block "
            << "character <.  Look above this line for the error:" << std::endl
            << line << std::endl
            << std::endl;
        PARTHENON_FAIL(msg);
      }
      first_char++;
      last_char = (line.find_first_of(">", first_char));
      block_name.assign(line, first_char, last_char - 1); // extract block name
      block_name = Path_(block_name, "");

      if (last_char == std::string::npos) {
        msg << "### FATAL ERROR in function [ParameterInput::LegacyParse]" << std::endl
            << "Block name '" << block_name << "' in the input stream'"
            << "' not properly ended";
        PARTHENON_FAIL(msg);
      }

      blocks_found++;
      continue; // skip to next line if block name was found
    } // end "a new block was found"

    // if line does not contain a block name or skippable information (comments,
    // whitespace), it must contain a parameter value
    if (blocks_found == 0) {
      msg << "### FATAL ERROR in function [ParameterInput::LegacyParse]" << std::endl
          << "Input file must specify a block name before the first"
          << " parameter = value line";
      PARTHENON_FAIL(msg);
    }
    // parse line and add name/value/comment strings (if found) to current block name
    bool has_cont_char = LegacyParseLine(line, param_name, param_value);
    if (continuing || has_cont_char) {
      // Append line data
      multiline_name += param_name;
      multiline_value += param_value;
      // Set new state
      continuing = true;
    }

    if (continuing && !has_cont_char) {
      // Flush line data
      param_name = multiline_name;
      param_value = multiline_value;
      param_comment = multiline_comment;
      multiline_name = "";
      multiline_value = "";
      multiline_comment = "";
      // Set new state
      continuing = false;
    }

    if (!continuing) {
      if (param_name != "") {
        toml::table single_param = toml::table();
        // std::cerr << "Existing table:" << std::endl << tmp_tbl;
        // std::cerr << "Adding " << Path_(block_name, param_name) << " from file " <<
        // fname << std::endl;
        AddParameter_(tmp_tbl, Path_(block_name, param_name), param_value,
                      ParameterOrigin::input, true, fname);
        // std::cerr << "Added" << std::endl;
      }
    }
  }

  return tmp_tbl;
}

bool ParameterInput::LegacyParseLine(std::string line, std::string &name,
                                     std::string &value) {
  std::size_t first_char, last_char, equal_char, hash_char, cont_char, len;
  bool continuation = false;

  hash_char = line.find_first_of("#"); // find "#" (optional)
  if (hash_char != std::string::npos) {
    line.erase(hash_char, std::string::npos);
  }

  first_char = line.find_first_not_of(" "); // find first non-white space
  equal_char = line.find_first_of("=");     // find "=" char

  // copy substring into name, remove white space at end of name
  if (equal_char == std::string::npos) {
    name = "";
    line.erase(0, first_char);
  } else {
    len = equal_char - first_char;
    name.assign(line, first_char, len);
    last_char = name.find_last_not_of(" ");
    name.erase(last_char + 1, std::string::npos);
    line.erase(0, equal_char + 1);
  }

  cont_char = line.find_first_of("&"); // find "&" continuation character
  // copy substring into value, remove white space at start and end
  len = cont_char;
  if (cont_char != std::string::npos) {
    std::string right_of_cont;
    right_of_cont.assign(line, cont_char + 1, std::string::npos);
    first_char = right_of_cont.find_first_not_of(" ");
    if (first_char != std::string::npos) {
      throw std::runtime_error("ERROR: Non-comment characters are not permitted to the "
                               "right of line continuations");
    }
    continuation = true;
  }
  value.assign(line, 0, len);

  first_char = value.find_first_not_of(" ");
  value.erase(0, first_char);

  last_char = value.find_last_not_of(" ");
  value.erase(last_char + 1, std::string::npos);

  return continuation;
}

} // namespace parthenon
