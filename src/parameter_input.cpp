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
//! \file parameter_input.cpp
//  \brief implementation of functions in class ParameterInput
//
// PURPOSE: Member functions of this class are used to read and parse the input file.
//   Functionality is loosely modeled after FORTRAN namelist.
//
// NOTE: This describes the standard text-based input format. Parthenon's architecture
//   supports multiple input formats that populate the same underlying parameter storage.
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
// This file was made in part with generative AI.

#include "parameter_input.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "globals.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {

//----------------------------------------------------------------------------------------
// ParameterInput constructor

ParameterInput::ParameterInput() : last_filename_{} {}

ParameterInput::ParameterInput(std::string input_filename) : last_filename_{} {
  IOWrapper infile;
  infile.Open(input_filename.c_str(), IOWrapper::FileMode::read);
  LoadFromFile(infile);
  infile.Close();
}

ParameterInput::~ParameterInput() = default;

//----------------------------------------------------------------------------------------
//! \fn  void ParameterInput::LoadFromStream(std::istream &is)
//  \brief Load input parameters from a stream

void ParameterInput::LoadFromStream(std::istream &is) {
  PARTHENON_REQUIRE_THROWS(!parsing_finalized_,
                           "Can't add new parameters after parsing is resolved.");
  std::string line, block_name, param_name, param_value, param_comment;
  std::size_t first_char, last_char;
  std::stringstream msg;
  int line_num{-1}, blocks_found{0};

  // Buffer multiple lines if a continuation character is present
  std::string multiline_name, multiline_value, multiline_comment;
  // Status in/out of continuation
  bool continuing = false;

  while (is.good()) {
    std::getline(is, line);
    line_num++;

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
        msg << "### FATAL ERROR in function [ParameterInput::LoadFromStream]" << std::endl
            << "Multiline field ended unexpectedly with new block "
            << "character <.  Look above this line for the error:" << std::endl
            << line << std::endl
            << std::endl;
        PARTHENON_FAIL(msg);
      }
      first_char++;
      last_char = (line.find_first_of(">", first_char));
      block_name.assign(line, first_char, last_char - 1); // extract block name

      if (last_char == std::string::npos) {
        msg << "### FATAL ERROR in function [ParameterInput::LoadFromStream]" << std::endl
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
      msg << "### FATAL ERROR in function [ParameterInput::LoadFromStream]" << std::endl
          << "Input file must specify a block name before the first"
          << " parameter = value line";
      PARTHENON_FAIL(msg);
    }
    // parse line and add name/value/comment strings (if found) to current block name
    bool has_cont_char = ParseLine(line, param_name, param_value, param_comment);
    if (continuing || has_cont_char) {
      // Append line data
      multiline_name += param_name;
      multiline_value += param_value;
      multiline_comment += param_comment;
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
        AddParsedParameter(block_name, param_name, UnresolvedString(param_value),
                           param_comment);
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn  void ParameterInput::LoadFromFile(IOWrapper &input)
//  \brief Read the parameters from an input file or restarting file.
//         Return the position at the end of the header, which is used in restarting

void ParameterInput::LoadFromFile(IOWrapper &input) {
  PARTHENON_REQUIRE_THROWS(
      !parsing_finalized_,
      "Can't add new parameters to the linked list after the map is resolved.");
  std::stringstream par, msg;
  constexpr int kBufSize = 4096;
  char buf[kBufSize];
  IOWrapperSizeT header = 0, ret, loc;

  // search <par_end> or EOF.
  do {
    if (Globals::my_rank == 0) // only the master process reads the header from the file
      ret = input.Read(buf, sizeof(char), kBufSize);
#ifdef MPI_PARALLEL
    // then broadcasts it
    PARTHENON_MPI_CHECK(
        MPI_Bcast(&ret, sizeof(IOWrapperSizeT), MPI_BYTE, 0, MPI_COMM_WORLD));
    PARTHENON_MPI_CHECK(MPI_Bcast(buf, ret, MPI_BYTE, 0, MPI_COMM_WORLD));
#endif
    par.write(buf, ret); // add the buffer into the stream
    header += ret;
    std::string sbuf = par.str();    // create string for search
    loc = sbuf.find("<par_end>", 0); // search from the top of the stream
    if (loc != std::string::npos) {  // found <par_end>
      header = loc + 10;             // store the header length
      break;
    }
    if (header > kBufSize * 10) {
      msg << "### FATAL ERROR in function [ParameterInput::LoadFromFile]"
          << "<par_end> is not found in the first 40KBytes." << std::endl
          << "Probably the file is broken or a wrong file is specified" << std::endl;
      PARTHENON_FAIL(msg);
    }
  } while (ret == kBufSize); // till EOF (or par_end is found)

  // Now par contains the parameter inputs + some additional including <par_end>
  // Read the stream and load the parameters
  LoadFromStream(par);
  // Seek the file to the end of the header
  input.Seek(header);

  return;
}

//----------------------------------------------------------------------------------------
//! \fn Block* ParameterInput::FindBlock_(const std::string & name)
//  \brief find specified Block.  Returns pointer to block or nullptr.

Block *ParameterInput::FindBlock_(const std::string &name) {
  auto it = block_index_.find(name);
  return (it != block_index_.end()) ? &param_storage_[it->second] : nullptr;
}

const Block *ParameterInput::FindBlock_(const std::string &name) const {
  auto it = block_index_.find(name);
  return (it != block_index_.end()) ? &param_storage_[it->second] : nullptr;
}

//----------------------------------------------------------------------------------------
//! \fn Block* ParameterInput::FindOrAddBlock_(const std::string & name)
//  \brief find or add specified Block.  Returns pointer to existing block if found,
//         or creates new block if not found. This allows parameters from
//         multiple sources (files, command line, Python, etc.) to populate
//         the same logical block.

Block *ParameterInput::FindOrAddBlock_(const std::string &name) {
  // Fast path: Check map first
  auto map_it = block_index_.find(name);
  if (map_it != block_index_.end()) {
    return &param_storage_[map_it->second]; // Block exists, return pointer using index
  }

  // Not found - create new block in vector and index it
  size_t new_idx = param_storage_.size();
  param_storage_.emplace_back(
      Block{name, {}, {}});     // name, params vector, param_index map
  block_index_[name] = new_idx; // Index it
  return &param_storage_[new_idx];
}

//----------------------------------------------------------------------------------------
//! \fn void ParameterInput::ParseLine(std::string line,
//           std::string& name, std::string& value, std::string& comment)
//  \brief parse "name = value # comment" format, return name/value/comment strings.

bool ParameterInput::ParseLine(std::string line, std::string &name, std::string &value,
                               std::string &comment) {
  std::size_t first_char, last_char, equal_char, hash_char, cont_char, len;
  bool continuation = false;

  hash_char = line.find_first_of("#"); // find "#" (optional)
  comment = "";
  if (hash_char != std::string::npos) {
    comment = line.substr(hash_char);
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

//----------------------------------------------------------------------------------------
//! void ParameterInput::ModifyFromCmdline(int argc, char *argv[])
//  \brief parse commandline for changes to input parameters
// Note this function is very forgiving (no warnings!) if there is an error in format

void ParameterInput::ModifyFromCmdline(int argc, char *argv[]) {
  PARTHENON_REQUIRE_THROWS(
      !parsing_finalized_,
      "Can't add new parameters to the linked list after the map is resolved.");
  std::string input_text, block, name, value;
  std::stringstream msg;

  for (int i = 1; i < argc; i++) {
    input_text = argv[i];
    std::size_t equal_posn = input_text.find_first_of("=");     // first "=" character
    std::size_t slash_posn = input_text.rfind("/", equal_posn); // last "/" before "="

    // skip if either "/" or "=" do not exist in input
    if ((slash_posn == std::string::npos) || (equal_posn == std::string::npos)) continue;

    if (slash_posn > equal_posn) {
      msg << "'/' used as value (rhs of =) when modifying " << input_text << "."
          << " Please update value of change "
          << "logic in ModifyFromCmdline function.";
      PARTHENON_FAIL(msg.str().c_str());
    }

    // extract block/name/value strings
    block = input_text.substr(0, slash_posn);
    name = input_text.substr(slash_posn + 1, (equal_posn - slash_posn - 1));
    value = input_text.substr(equal_posn + 1, std::string::npos);

    // Check if block/parameter exists for warning messages
    Block *pb = FindBlock_(block);
    if (pb == nullptr) {
      if (Globals::my_rank == 0) {
        msg << "In function [ParameterInput::ModifyFromCmdline]:" << std::endl
            << "               Block name '" << block
            << "' on command line not found in input/restart file. Block will be added.";
        PARTHENON_WARN(msg);
      }
    } else if (FindParameter_(block, name) == nullptr) {
      if (Globals::my_rank == 0) {
        msg << "In function [ParameterInput::ModifyFromCmdline]:" << std::endl
            << "               Parameter '" << name << "' in block '" << block
            << "' on command line not found in input/restart file. Parameter will be "
               "added.";
        PARTHENON_WARN(msg);
      }
    }

    // Add or update parameter (handles both map and linked list)
    AddParsedParameter(block, name, UnresolvedString(value), "# From command line");
  }
}

//----------------------------------------------------------------------------------------
//! \fn bool ParameterInput::DoesParameterExist(const std::string & block, const
//! std::string & name)
//  \brief check whether parameter of given name in given block exists

bool ParameterInput::DoesParameterExist(const std::string &block,
                                        const std::string &name) {
  return FindParameter_(block, name) != nullptr;
}

//----------------------------------------------------------------------------------------
//! \fn bool ParameterInput::DoesBlockExist(const std::string & block)
//  \brief check whether block exists

bool ParameterInput::DoesBlockExist(const std::string &block) {
  return FindBlock_(block) != nullptr;
}

std::string ParameterInput::GetComment(const std::string &block,
                                       const std::string &name) {
  std::stringstream msg;

  const Parameter *param = FindParameter_(block, name);
  if (param == nullptr) {
    msg << "### FATAL ERROR in function [ParameterInput::GetComment]" << std::endl
        << "Parameter '" << name << "' not found in block '" << block << "'";
    PARTHENON_FAIL(msg);
  }

  return param->comment;
}

//----------------------------------------------------------------------------------------
//! \fn std::string ParameterInput::GetAsUnresolvedString(const std::string & block,
//! const std::string & name)
//  \brief returns string representation of parameter value, preferring original string
//  from input file when available

std::string ParameterInput::GetAsUnresolvedString(const std::string &block,
                                                  const std::string &name) {
  FinalizeParsing(); // Ensure parsing is complete (consistent with other getters)

  const Parameter *param = FindParameter_(block, name);
  if (param == nullptr) {
    std::stringstream msg;
    msg << "### FATAL ERROR in function [ParameterInput::GetAsUnresolvedString]"
        << std::endl
        << "Parameter name '" << name << "' not found in block '" << block << "'";
    PARTHENON_THROW(msg);
  }
  return param->ToString();
}

//----------------------------------------------------------------------------------------
//! \fn int ParameterInput::GetInteger(const std::string & block, const std::string &
//! name)
//  \brief returns integer value of string stored in block/name

int ParameterInput::GetInteger(const std::string &block, const std::string &name,
                               const std::optional<std::string> &docstring) {
  return Get<int>(block, name, docstring);
}

//----------------------------------------------------------------------------------------
//! \fn Real ParameterInput::GetReal(const std::string & block, const std::string & name)
//  \brief returns real value of string stored in block/name

Real ParameterInput::GetReal(const std::string &block, const std::string &name,
                             const std::optional<std::string> &docstring) {
  return Get<Real>(block, name, docstring);
}

//----------------------------------------------------------------------------------------
//! \fn bool ParameterInput::GetBoolean(const std::string & block, const std::string &
//! name)
//  \brief returns boolean value of string stored in block/name

bool ParameterInput::GetBoolean(const std::string &block, const std::string &name,
                                const std::optional<std::string> &docstring) {
  return Get<bool>(block, name, docstring);
}

//----------------------------------------------------------------------------------------
//! \fn std::string ParameterInput::GetString(const std::string & block, const std::string
//! & name)
//  \brief returns string stored in block/name

std::string ParameterInput::GetString(const std::string &block, const std::string &name,
                                      const std::optional<std::string> &docstring) {
  return Get<std::string>(block, name, docstring);
}

std::string ParameterInput::GetString(const std::string &block, const std::string &name,
                                      const std::vector<std::string> &allowed_values,
                                      const std::optional<std::string> &docstring) {
  auto val = GetString(block, name);
  CheckAllowedValues_(block, name, val, allowed_values);
  CheckAndUpdateQueries_<std::string>(block, name, std::optional<std::string>{},
                                      allowed_values, docstring);
  return val;
}

//----------------------------------------------------------------------------------------
//! \fn int ParameterInput::GetOrAddInteger(const std::string & block, const std::string &
//! name,
//    int default_value)
//  \brief returns integer value stored in block/name if it exists, or creates and sets
//  value to def_value if it does not exist

int ParameterInput::GetOrAddInteger(const std::string &block, const std::string &name,
                                    int def_value,
                                    const std::optional<std::string> &docstring) {
  return GetOrAdd<int>(block, name, def_value, docstring);
}
int ParameterInput::GetOrAddInteger(const std::string &block, const std::string &name,
                                    const ParameterRef &value,
                                    const std::optional<std::string> &docstring) {
  auto defval = Get<int>(value);
  auto ret = GetOrAddInteger(block, name, defval, docstring);
  SetQueryDependency_(block, name, value);
  return ret;
}

//----------------------------------------------------------------------------------------
//! \fn Real ParameterInput::GetOrAddReal(const std::string & block, const std::string &
//! name,
//    Real def_value)
//  \brief returns real value stored in block/name if it exists, or creates and sets
//  value to def_value if it does not exist

Real ParameterInput::GetOrAddReal(const std::string &block, const std::string &name,
                                  Real def_value,
                                  const std::optional<std::string> &docstring) {
  return GetOrAdd<Real>(block, name, def_value, docstring);
}
Real ParameterInput::GetOrAddReal(const std::string &block, const std::string &name,
                                  const ParameterRef &value,
                                  const std::optional<std::string> &docstring) {
  auto defval = Get<Real>(value);
  auto ret = GetOrAddReal(block, name, defval, docstring);
  SetQueryDependency_(block, name, value);
  return ret;
}

//----------------------------------------------------------------------------------------
//! \fn bool ParameterInput::GetOrAddBoolean(const std::string & block, const std::string
//! & name,
//    bool def_value)
//  \brief returns boolean value stored in block/name if it exists, or creates and sets
//  value to def_value if it does not exist

bool ParameterInput::GetOrAddBoolean(const std::string &block, const std::string &name,
                                     bool def_value,
                                     const std::optional<std::string> &docstring) {
  return GetOrAdd<bool>(block, name, def_value, docstring);
}
bool ParameterInput::GetOrAddBoolean(const std::string &block, const std::string &name,
                                     const ParameterRef &value,
                                     const std::optional<std::string> &docstring) {
  auto defval = Get<bool>(value);
  auto ret = GetOrAddBoolean(block, name, defval, docstring);
  SetQueryDependency_(block, name, value);
  return ret;
}

//----------------------------------------------------------------------------------------
//! \fn std::string ParameterInput::GetOrAddString(const std::string & block, const
//! std::string & name,
//                                                 const std::string & def_value)
//  \brief returns string value stored in block/name if it exists, or creates and sets
//  value to def_value if it does not exist

std::string ParameterInput::GetOrAddString(const std::string &block,
                                           const std::string &name,
                                           const std::string &def_value,
                                           const std::optional<std::string> &docstring) {
  return GetOrAdd<std::string>(block, name, def_value, docstring);
}

std::string ParameterInput::GetOrAddString(const std::string &block,
                                           const std::string &name,
                                           const std::string &def_value,
                                           const std::vector<std::string> &allowed_values,
                                           const std::optional<std::string> &docstring) {
  auto val = GetOrAddString(block, name, def_value);
  CheckAllowedValues_(block, name, val, allowed_values);
  CheckAndUpdateQueries_<std::string>(block, name, def_value, allowed_values, docstring);
  return val;
}

//----------------------------------------------------------------------------------------
//! \fn int ParameterInput::SetInteger(const std::string & block, const std::string &
//! name, int value)
//  \brief updates an integer parameter; creates it if it does not exist

int ParameterInput::SetInteger(const std::string &block, const std::string &name,
                               int value, const std::optional<std::string> &docstring) {
  return Set<int>(block, name, value, docstring);
}

//----------------------------------------------------------------------------------------
//! \fn Real ParameterInput::SetReal(const std::string & block, const std::string &
//! name, Real value)
//  \brief updates a real parameter with full precision; creates it if it does not exist

Real ParameterInput::SetReal(const std::string &block, const std::string &name,
                             Real value, const std::optional<std::string> &docstring) {
  return Set<Real>(block, name, value, docstring);
}

//----------------------------------------------------------------------------------------
//! \fn bool ParameterInput::SetBoolean(const std::string & block, const std::string &
//! name, bool value)
//  \brief updates a boolean parameter; creates it if it does not exist

bool ParameterInput::SetBoolean(const std::string &block, const std::string &name,
                                bool value, const std::optional<std::string> &docstring) {
  return Set<bool>(block, name, value, docstring);
}

//----------------------------------------------------------------------------------------
//! \fn std::string ParameterInput::SetString(const std::string & block, const std::string
//! & name,
//                                            std::string  value)
//  \brief updates a string parameter; creates it if it does not exist

std::string ParameterInput::SetString(const std::string &block, const std::string &name,
                                      const std::string &value,
                                      const std::optional<std::string> &docstring) {
  return Set<std::string>(block, name, value, docstring);
}

void ParameterInput::RemoveParameter(const std::string &block, const std::string &name) {
  // Remove from storage
  Block *pb = FindBlock_(block);
  if (pb != nullptr) {
    auto it = std::remove_if(pb->params.begin(), pb->params.end(),
                             [&name](const Parameter &p) { return p.name == name; });
    pb->params.erase(it, pb->params.end());

    // Rebuild param_index since indices have shifted
    pb->param_index.clear();
    for (size_t i = 0; i < pb->params.size(); ++i) {
      pb->param_index[pb->params[i].name] = i;
    }
  }

  // Remove from query records
  auto key = std::make_pair(block, name);
  queries_.erase(key);
}

void ParameterInput::CheckRequired(const std::string &block, const std::string &name) {
  bool missing = true;
  if (DoesParameterExist(block, name)) {
    missing = (GetComment(block, name) == "# Default value added at run time");
  }
  if (missing) {
    std::stringstream ss;
    ss << std::endl
       << "### ERROR in CheckRequired:" << std::endl
       << "Parameter file missing required field <" << block << ">/" << name << std::endl
       << std::endl;
    throw std::runtime_error(ss.str());
  }
}

void ParameterInput::CheckDesired(const std::string &block, const std::string &name) {
  bool missing = true;
  bool defaulted = false;
  if (DoesParameterExist(block, name)) {
    missing = false;
    defaulted = (GetComment(block, name) == "# Default value added at run time");
  }
  if (missing) {
    std::cout << std::endl
              << "### WARNING in CheckDesired:" << std::endl
              << "Parameter file missing desired field <" << block << ">/" << name
              << std::endl;
  }
  if (defaulted) {
    auto *param = FindParameter_(block, name);
    std::cout << std::endl
              << "Defaulting to <" << block << ">/" << name << " = " << param->ToString()
              << std::endl;
  }
}

void ParameterInput::CheckOrphans() const {
  std::set<std::pair<std::string, std::string>> orphans;
  for (const auto &block : param_storage_) {
    for (const auto &param : block.params) {
      auto key = std::make_pair(block.name, param.name);
      if (queries_.count(key) == 0) {
        orphans.insert(key);
      }
    }
  }
  std::stringstream msg;
  msg << "The following input parameters are set but unused:\n";
  for (const auto &[b, p] : orphans) {
    msg << b << "/" << p << "\n";
  }
  msg << std::endl;
  PARTHENON_WARN(msg);
}

//----------------------------------------------------------------------------------------
//! \fn void ParameterInput::ParameterDump(std::ostream& os)
//  \brief output entire parameter storage to specified stream

void ParameterInput::ParameterDump(std::ostream &os) {
  os << "#------------------------- PAR_DUMP -------------------------" << std::endl;

  for (const auto &block : param_storage_) {
    os << "<" << block.name << ">" << std::endl;

    // Find max lengths for alignment
    std::size_t max_len_name = 0;
    std::size_t max_len_value = 0;
    for (const auto &param : block.params) {
      std::string value_str = param.ToString();
      max_len_name = std::max(max_len_name, param.name.length());
      max_len_value = std::max(max_len_value, value_str.length());
    }

    // Output parameters with alignment
    for (const auto &param : block.params) {
      std::string param_name = param.name;
      auto key = std::make_pair(block.name, param.name);
      auto record_it = queries_.find(key);
      std::string param_value = param.ToString();

      if (record_it != queries_.end() && record_it->second.IsDefaultEmptyStringVec() &&
          param_value.empty()) {
        continue;
      }

      std::size_t len = max_len_name - param_name.length() + 1;
      param_name.append(len, ' '); // pad name to align vertically
      len = max_len_value - param_value.length() + 1;
      param_value.append(len, ' '); // pad value to align vertically

      os << param_name << "= " << param_value << param.comment << std::endl;
    }
  }

  os << "#------------------------- PAR_DUMP -------------------------" << std::endl;
  os << "<par_end>" << std::endl; // finish with par-end (useful in restart files)
}

void ParameterInput::OutputParameterTable(std::ostream &os,
                                          const std::regex &block_regex) const {
  // Loop through once and store in a map for lexicographic ordering
  std::map<std::string, std::map<std::string, std::string>> csvblocks;
  for (const auto &block : param_storage_) {
    const std::string &block_name = block.name;
    if (std::regex_match(block_name, block_regex)) {
      auto &csvlines = csvblocks[block_name];
      for (const auto &param : block.params) {
        const std::string &param_name = param.name;
        auto record_key = std::make_pair(block_name, param_name);
        /* clang-format off */
        // This ensures the code doesn't crash for orphan parameters
        if (queries_.count(record_key) > 0) {
          auto record = queries_.at(record_key);
          std::stringstream ss;
          ss << "\"" << block_name << "\""
             << "," << "\"" << param_name << "\""
             << "," << "\"" << record.param_type << "\""
             << "," << "\"" << record.default_value_str << "\""
             << "," << "\"";
          std::size_t num_allowed_vals = record.allowed_vals_str.size();
          if (record.docstring.has_value()) {
            ss << record.docstring.value();
            if (num_allowed_vals > 0) {
              ss << "; ";
            }
          }
          if (num_allowed_vals > 0) {
            ss << "Allowed values: ";
            std::size_t ival = 0;
            for (const auto &v : record.allowed_vals_str) {
              ss << v;
              if (ival < num_allowed_vals - 1) {
                ss << ", ";
              }
              ival++;
            }
          }
          ss << "\"";
          /* clang-format on */
          csvlines[param_name] = ss.str();
        }
      }
    }
  }

  os << "block,parameters,type,default,description" << std::endl;
  int i = 0;
  for (const auto &[bname, b] : csvblocks) {
    if (b.size() > 0) { // special case for empty block
      // An empty row to demark blocks. Can be filtered on by a parser
      // or grep
      if (i != 0) {
        os << "\"\",\"\",\"\",\"\",\"\"" << std::endl;
      }
      for (const auto &[p, l] : b) {
        os << l << std::endl;
      }
      i++;
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn std::string Param::ToString()
//  \brief Convert a ParamValue variant to string for linked list output

std::string Parameter::ToString() const {
  std::stringstream ss;
  if (original_string.has_value()) {
    return original_string.value().value;
  }

  if (std::holds_alternative<UnresolvedString>(value)) {
    ss << std::get<UnresolvedString>(value).value;
  } else if (std::holds_alternative<int>(value)) {
    ss << std::get<int>(value);
  } else if (std::holds_alternative<Real>(value)) {
    ss.precision(std::numeric_limits<Real>::max_digits10);
    ss << std::get<Real>(value);
  } else if (std::holds_alternative<bool>(value)) {
    ss << (std::get<bool>(value) ? "true" : "false");
  } else if (std::holds_alternative<std::string>(value)) {
    ss << std::get<std::string>(value);
  } else if (std::holds_alternative<std::vector<int>>(value)) {
    const auto &vec = std::get<std::vector<int>>(value);
    for (size_t i = 0; i < vec.size(); ++i) {
      if (i > 0) ss << ", ";
      ss << vec[i];
    }
  } else if (std::holds_alternative<std::vector<Real>>(value)) {
    const auto &vec = std::get<std::vector<Real>>(value);
    ss.precision(std::numeric_limits<Real>::max_digits10);
    for (size_t i = 0; i < vec.size(); ++i) {
      if (i > 0) ss << ", ";
      ss << vec[i];
    }
  } else if (std::holds_alternative<std::vector<bool>>(value)) {
    const auto &vec = std::get<std::vector<bool>>(value);
    for (size_t i = 0; i < vec.size(); ++i) {
      if (i > 0) ss << ", ";
      ss << (vec[i] ? "true" : "false");
    }
  } else if (std::holds_alternative<std::vector<std::string>>(value)) {
    const auto &vec = std::get<std::vector<std::string>>(value);
    for (size_t i = 0; i < vec.size(); ++i) {
      if (i > 0) ss << ", ";
      ss << vec[i];
    }
  }

  return ss.str();
}

//----------------------------------------------------------------------------------------
//! \fn void ParameterInput::AddParameter_()
//  \brief Internal helper to add/update parameter without resolution check
//  Used by GetOrAdd/Set (which need to add defaults after resolution)
//  and by AddParsedParameter (which enforces the resolution check for parsers)

void ParameterInput::AddParameter_(const std::string &block, const std::string &name,
                                   const ParamValue &value, const std::string &comment) {
  // Find or add the block
  Block *pb = FindOrAddBlock_(block);

  // Check if parameter already exists in this block using the index
  auto param_it = pb->param_index.find(name);
  if (param_it != pb->param_index.end()) {
    // Parameter exists - update value and comment
    size_t idx = param_it->second;
    pb->params[idx].value = value;
    pb->params[idx].comment = comment;
    return;
  }

  // Parameter doesn't exist - add new one
  size_t new_idx = pb->params.size();
  pb->params.emplace_back(Parameter{name, comment, value});
  pb->param_index[name] = new_idx; // Index it
}

//----------------------------------------------------------------------------------------
//! \fn void ParameterInput::AddParsedParameter()
//  \brief Public interface for parsers to add parameters to storage
//  Can be called by any parser (text, Python, TOML, etc.) to populate param_storage_
//  Enforces that parsing must not be resolved yet.

void ParameterInput::AddParsedParameter(const std::string &block, const std::string &name,
                                        const ParamValue &value,
                                        const std::string &comment) {
  PARTHENON_REQUIRE_THROWS(!parsing_finalized_,
                           "Can't add new parameters after parsing is resolved.");
  AddParameter_(block, name, value, comment);
}

//----------------------------------------------------------------------------------------
//! \fn void ParameterInput::FinalizeParsing()
//  \brief Finalize the parsing phase - no more parsing allowed (GetOrAdd/Set still work)

void ParameterInput::FinalizeParsing() { parsing_finalized_ = true; }

//----------------------------------------------------------------------------------------
//! \fn std::vector<std::string> ParameterInput::GetBlockNames()
//  \brief Return all block names in the input

std::vector<std::string> ParameterInput::GetBlockNames() const {
  std::vector<std::string> names;
  names.reserve(param_storage_.size());
  for (const auto &block : param_storage_) {
    names.push_back(block.name);
  }
  return names;
}

//----------------------------------------------------------------------------------------
//! \fn std::vector<std::string> ParameterInput::GetBlockNamesWithPrefix()
//  \brief Return all block names that start with the given prefix

std::vector<std::string>
ParameterInput::GetBlockNamesWithPrefix(const std::string &prefix) const {
  std::vector<std::string> matching_blocks;

  for (const auto &block : param_storage_) {
    if (block.name.compare(0, prefix.length(), prefix) == 0) {
      matching_blocks.push_back(block.name);
    }
  }

  return matching_blocks;
}

//----------------------------------------------------------------------------------------
//! \fn std::vector<std::string> ParameterInput::GetParameterNames()
//  \brief Return all parameter names in the given block

std::vector<std::string>
ParameterInput::GetParameterNames(const std::string &block) const {
  std::vector<std::string> param_names;

  const Block *pb = FindBlock_(block);
  if (pb != nullptr) {
    param_names.reserve(pb->params.size());
    for (const auto &param : pb->params) {
      param_names.push_back(param.name);
    }
  }

  return param_names;
}

//----------------------------------------------------------------------------------------
//! \fn Parameter* ParameterInput::FindParameter_()
//  \brief Helper to find a parameter in storage

Parameter *ParameterInput::FindParameter_(const std::string &block,
                                          const std::string &name) {
  Block *pb = FindBlock_(block);
  if (pb == nullptr) return nullptr;

  auto it = pb->param_index.find(name);
  return (it != pb->param_index.end()) ? &pb->params[it->second] : nullptr;
}

const Parameter *ParameterInput::FindParameter_(const std::string &block,
                                                const std::string &name) const {
  const Block *pb = FindBlock_(block);
  if (pb == nullptr) return nullptr;

  auto it = pb->param_index.find(name);
  return (it != pb->param_index.end()) ? &pb->params[it->second] : nullptr;
}

//----------------------------------------------------------------------------------------
//! \fn template <typename T> std::optional<T> ParameterInput::GetFromStorage_()
//  \brief Helper to get a typed parameter from storage with caching
//  Returns nullopt if not in storage, throws on type mismatch

template <typename T>
std::optional<T> ParameterInput::GetFromStorage_(const std::string &block,
                                                 const std::string &name) {
  FinalizeParsing();
  Parameter *param = FindParameter_(block, name);
  if (param == nullptr) {
    return std::nullopt; // Not in storage
  }

  // If it's an UnresolvedString, convert and cache
  if (std::holds_alternative<UnresolvedString>(param->value)) {
    if (!param->original_string.has_value()) {
      param->original_string = std::get<UnresolvedString>(param->value);
    }
    T typed_val = ConvertParamValue<T>(param->value, block, name);
    param->value = typed_val; // Cache the typed value in the variant
    return typed_val;
  }

  // If it's already the correct type, return it
  if (std::holds_alternative<T>(param->value)) {
    return std::get<T>(param->value);
  }

  // Type mismatch - was previously resolved as a different type
  std::stringstream msg;
  msg << "### FATAL ERROR in ParameterInput::GetFromStorage_" << std::endl
      << "Parameter '" << name << "' in block '" << block
      << "' was previously accessed as a different type" << std::endl;
  PARTHENON_FAIL(msg);
}

//----------------------------------------------------------------------------------------
//! \fn std::vector<std::string> ParameterInput::SplitCommaSeparated()
//  \brief Helper to split comma-separated values

std::vector<std::string> ParameterInput::SplitCommaSeparated(const std::string &s) {
  std::string str = s;
  std::string delimiter = ",";
  size_t pos = 0;
  std::string token;
  std::vector<std::string> variables;

  while ((pos = str.find(delimiter)) != std::string::npos) {
    token = str.substr(0, pos);
    variables.push_back(string_utils::trim(token));
    str.erase(0, pos + delimiter.length());
  }
  token = string_utils::trim(str);
  if (!token.empty()) {
    variables.push_back(token);
  }

  return variables;
}

//----------------------------------------------------------------------------------------
//! \fn template <typename T> T ParameterInput::ConvertParamValue()
//  \brief Convert a ParamValue variant to the requested type

template <typename T>
T ParameterInput::ConvertParamValue(const ParamValue &value, const std::string &block,
                                    const std::string &name) {
  std::stringstream msg;

  // If it's already the right type, return it
  if (std::holds_alternative<T>(value)) {
    return std::get<T>(value);
  }

  // If it's an unresolved string, convert it
  if (std::holds_alternative<UnresolvedString>(value)) {
    const std::string &str_val = std::get<UnresolvedString>(value).value;

    constexpr bool is_vector_type = std::is_same_v<T, std::vector<int>> ||
                                    std::is_same_v<T, std::vector<Real>> ||
                                    std::is_same_v<T, std::vector<bool>> ||
                                    std::is_same_v<T, std::vector<std::string>>;

    if constexpr (std::is_same_v<T, int>) {
      return stoi(str_val);
    } else if constexpr (std::is_same_v<T, Real>) {
      return static_cast<Real>(atof(str_val.c_str()));
    } else if constexpr (std::is_same_v<T, bool>) {
      return stob(str_val);
    } else if constexpr (std::is_same_v<T, std::string>) {
      return str_val;
    } else if constexpr (is_vector_type) {
      using ElemType = typename T::value_type;
      std::vector<std::string> fields = SplitCommaSeparated(str_val);
      T result;

      for (const auto &field : fields) {
        if constexpr (std::is_same_v<ElemType, int>) {
          result.push_back(stoi(field));
        } else if constexpr (std::is_same_v<ElemType, Real>) {
          result.push_back(static_cast<Real>(atof(field.c_str())));
        } else if constexpr (std::is_same_v<ElemType, bool>) {
          result.push_back(stob(field));
        } else if constexpr (std::is_same_v<ElemType, std::string>) {
          result.push_back(field);
        }
      }
      return result;
    }
  }

  msg << "### FATAL ERROR in function [ParameterInput::ConvertParamValue]" << std::endl
      << "Type mismatch for parameter '" << name << "' in block '" << block << "'"
      << std::endl;
  PARTHENON_FAIL(msg);
}

// Explicit template instantiations
template int ParameterInput::ConvertParamValue<int>(const ParamValue &,
                                                    const std::string &,
                                                    const std::string &);
template Real ParameterInput::ConvertParamValue<Real>(const ParamValue &,
                                                      const std::string &,
                                                      const std::string &);
template bool ParameterInput::ConvertParamValue<bool>(const ParamValue &,
                                                      const std::string &,
                                                      const std::string &);
template std::string ParameterInput::ConvertParamValue<std::string>(const ParamValue &,
                                                                    const std::string &,
                                                                    const std::string &);
template std::vector<int> ParameterInput::ConvertParamValue<std::vector<int>>(
    const ParamValue &, const std::string &, const std::string &);
template std::vector<Real> ParameterInput::ConvertParamValue<std::vector<Real>>(
    const ParamValue &, const std::string &, const std::string &);
template std::vector<bool> ParameterInput::ConvertParamValue<std::vector<bool>>(
    const ParamValue &, const std::string &, const std::string &);
template std::vector<std::string>
ParameterInput::ConvertParamValue<std::vector<std::string>>(const ParamValue &,
                                                            const std::string &,
                                                            const std::string &);

// Explicit instantiations for GetFromStorage_
template std::optional<int> ParameterInput::GetFromStorage_<int>(const std::string &,
                                                                 const std::string &);
template std::optional<Real> ParameterInput::GetFromStorage_<Real>(const std::string &,
                                                                   const std::string &);
template std::optional<bool> ParameterInput::GetFromStorage_<bool>(const std::string &,
                                                                   const std::string &);
template std::optional<std::string>
ParameterInput::GetFromStorage_<std::string>(const std::string &, const std::string &);
template std::optional<std::vector<int>>
ParameterInput::GetFromStorage_<std::vector<int>>(const std::string &,
                                                  const std::string &);
template std::optional<std::vector<Real>>
ParameterInput::GetFromStorage_<std::vector<Real>>(const std::string &,
                                                   const std::string &);
template std::optional<std::vector<bool>>
ParameterInput::GetFromStorage_<std::vector<bool>>(const std::string &,
                                                   const std::string &);
template std::optional<std::vector<std::string>>
ParameterInput::GetFromStorage_<std::vector<std::string>>(const std::string &,
                                                          const std::string &);

} // namespace parthenon
