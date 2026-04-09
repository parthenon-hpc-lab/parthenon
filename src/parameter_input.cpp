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
// NOTE: This describes the standard text-based input format. Parthenon also supports
//   other input formats (e.g., Python scripts via -i script.py) that populate the
//   same underlying parameter storage.
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

ParameterInput::ParameterInput() : pfirst_block{}, last_filename_{} {}

ParameterInput::ParameterInput(std::string input_filename)
    : pfirst_block{}, last_filename_{} {
  IOWrapper infile;
  infile.Open(input_filename.c_str(), IOWrapper::FileMode::read);
  LoadFromFile(infile);
  infile.Close();
}

// ParameterInput destructor- iterates through nested singly linked lists of blocks/lines
// and deletes each InputBlock node (whose destructor below deletes linked list "line"
// nodes)

ParameterInput::~ParameterInput() {
  InputBlock *pib = pfirst_block;
  while (pib != nullptr) {
    InputBlock *pold_block = pib;
    pib = pib->pnext;
    delete pold_block;
  }
}

// InputBlock destructor- iterates through singly linked list of "line" nodes and deletes
// them

InputBlock::~InputBlock() {
  InputLine *pil = pline;
  while (pil != nullptr) {
    InputLine *pold_line = pil;
    pil = pil->pnext;
    delete pold_line;
  }
}

//----------------------------------------------------------------------------------------
//! \fn  void ParameterInput::LoadFromStream(std::istream &is)
//  \brief Load input parameters from a stream

//  Input block names are allocated and stored in a singly linked list of InputBlocks.
//  Within each InputBlock the names, values, and comments of each parameter are allocated
//  and stored in a singly linked list of InputLines.

void ParameterInput::LoadFromStream(std::istream &is) {
  PARTHENON_REQUIRE(
      !map_resolved_,
      "Can't add new parameters to the linked list after the map is resolved.");
  std::string line, block_name, param_name, param_value, param_comment;
  std::size_t first_char, last_char;
  std::stringstream msg;
  InputBlock *pib{};
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

      pib = FindOrAddBlock(block_name); // find or add block to singly linked list

      if (pib == nullptr) {
        msg << "### FATAL ERROR in function [ParameterInput::LoadFromStream]" << std::endl
            << "Block name '" << block_name << "' could not be found/added";
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
    bool has_cont_char = ParseLine(pib, line, param_name, param_value, param_comment);
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
  PARTHENON_REQUIRE(
      !map_resolved_,
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
//! \fn InputBlock* ParameterInput::FindOrAddBlock(const std::string & name)
//  \brief find or add specified InputBlock.  Returns pointer to block.

InputBlock *ParameterInput::FindOrAddBlock(const std::string &name) {
  InputBlock *pib, *plast;
  plast = pfirst_block;
  pib = pfirst_block;

  // Search singly linked list of InputBlocks to see if name exists, return if found.
  while (pib != nullptr) {
    if (name.compare(pib->block_name) == 0) return pib;
    plast = pib;
    pib = pib->pnext;
  }

  // Create new block in list if not found above
  pib = new InputBlock;
  pib->block_name.assign(name); // store the new block name
  pib->pline = nullptr;         // Terminate the InputLine list
  pib->pnext = nullptr;         // Terminate the InputBlock list

  // Default max lengths to zero (in case of no parameters in this block)
  pib->max_len_parname = 0;
  pib->max_len_parvalue = 0;

  // if this is the first block in list, save pointer to it in class
  if (pfirst_block == nullptr) {
    pfirst_block = pib;
  } else {
    plast->pnext = pib; // link new node into list
  }

  return pib;
}

//----------------------------------------------------------------------------------------
//! \fn void ParameterInput::ParseLine(InputBlock *pib, std::string line,
//           std::string& name, std::string& value, std::string& comment)
//  \brief parse "name = value # comment" format, return name/value/comment strings.

bool ParameterInput::ParseLine(InputBlock *pib, std::string line, std::string &name,
                               std::string &value, std::string &comment) {
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
//! \fn void ParameterInput::AddParameter(InputBlock *pb, const std::string & name,
//   std::string value, const std::string & comment)
//  \brief add name/value/comment tuple to the InputLine singly linked list in block *pb.
//  If a parameter with the same name already exists, the value and comment strings
//  are replaced (overwritten).

void ParameterInput::AddParameter(InputBlock *pb, const std::string &name,
                                  const std::string &value, const std::string &comment) {
  InputLine *pl, *plast;
  // Search singly linked list of InputLines to see if name exists.  This also sets *plast
  // to point to the tail node (but not storing a pointer to the tail node in InputBlock)
  pl = pb->pline;
  plast = pb->pline;
  while (pl != nullptr) {
    if (name.compare(pl->param_name) == 0) { // param name already exists
      pl->param_value.assign(value);         // replace existing param value
      pl->param_comment.assign(comment);     // replace exisiting param comment
      if (value.length() > pb->max_len_parvalue) pb->max_len_parvalue = value.length();
      return;
    }
    plast = pl;
    pl = pl->pnext;
  }

  // Create new node in singly linked list if name does not already exist
  pl = new InputLine;
  pl->param_name.assign(name);
  pl->param_value.assign(value);
  pl->param_comment.assign(comment);
  pl->pnext = nullptr;

  // if this is the first parameter in list, save pointer to it in block.
  if (pb->pline == nullptr) {
    pb->pline = pl;
    pb->max_len_parname = name.length();
    pb->max_len_parvalue = value.length();
  } else {
    plast->pnext = pl; // link new node into list
    if (name.length() > pb->max_len_parname) pb->max_len_parname = name.length();
    if (value.length() > pb->max_len_parvalue) pb->max_len_parvalue = value.length();
  }

  return;
}

//----------------------------------------------------------------------------------------
//! void ParameterInput::ModifyFromCmdline(int argc, char *argv[])
//  \brief parse commandline for changes to input parameters
// Note this function is very forgiving (no warnings!) if there is an error in format

void ParameterInput::ModifyFromCmdline(int argc, char *argv[]) {
  PARTHENON_REQUIRE(
      !map_resolved_,
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

    // Check if block/parameter exists in map for warning messages
    auto block_it = param_map_.find(block);
    if (block_it == param_map_.end()) {
      if (Globals::my_rank == 0) {
        msg << "In function [ParameterInput::ModifyFromCmdline]:" << std::endl
            << "               Block name '" << block
            << "' on command line not found in input/restart file. Block will be added.";
        PARTHENON_WARN(msg);
      }
    } else {
      auto param_it = block_it->second.find(name);
      if (param_it == block_it->second.end()) {
        if (Globals::my_rank == 0) {
          msg << "In function [ParameterInput::ModifyFromCmdline]:" << std::endl
              << "               Parameter '" << name << "' in block '" << block
              << "' on command line not found in input/restart file. Parameter will be "
                 "added.";
          PARTHENON_WARN(msg);
        }
      }
    }

    // Add or update parameter (handles both map and linked list)
    AddParsedParameter(block, name, UnresolvedString(value), "# From command line");
  }
}

//----------------------------------------------------------------------------------------
//! \fn InputBlock* ParameterInput::GetPtrToBlock(const std::string & name)
//  \brief return pointer to specified InputBlock if it exists

InputBlock *ParameterInput::GetPtrToBlock(const std::string &name) {
  InputBlock *pb;
  for (pb = pfirst_block; pb != nullptr; pb = pb->pnext) {
    if (name.compare(pb->block_name) == 0) return pb;
  }
  return nullptr;
}

//----------------------------------------------------------------------------------------
//! \fn int ParameterInput::DoesParameterExist(const std::string & block, const
//! std::string & name)
//  \brief check whether parameter of given name in given block exists

int ParameterInput::DoesParameterExist(const std::string &block,
                                       const std::string &name) {
  MarkResolved();

  auto block_it = param_map_.find(block);
  if (block_it == param_map_.end()) return 0;

  auto param_it = block_it->second.find(name);
  return (param_it != block_it->second.end()) ? 1 : 0;
}

//----------------------------------------------------------------------------------------
//! \fn int ParameterInput::DoesBlockExist(const std::string & block)
//  \brief check whether block exists

int ParameterInput::DoesBlockExist(const std::string &block) {
  MarkResolved();

  auto block_it = param_map_.find(block);
  return (block_it != param_map_.end()) ? 1 : 0;
}

std::string ParameterInput::GetComment(const std::string &block,
                                       const std::string &name) {
  InputBlock *pb;
  InputLine *pl;
  std::stringstream msg;

  // get pointer to node with same block name in singly linked list of InputBlocks
  pb = GetPtrToBlock(block);
  if (pb == nullptr) {
    msg << "### FATAL ERROR in function [ParameterInput::GetComment]" << std::endl
        << "Block name '" << block << "' not found when trying to set value "
        << "for parameter '" << name << "'";
    PARTHENON_FAIL(msg);
  }

  // get pointer to node with same parameter name in singly linked list of InputLines
  pl = pb->GetPtrToLine(name);
  if (pl == nullptr) {
    msg << "### FATAL ERROR in function [ParameterInput::GetComment]" << std::endl
        << "Parameter name '" << name << "' not found in block '" << block << "'";
    PARTHENON_FAIL(msg);
  }

  std::string val = pl->param_comment;
  return val;
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
  // Remove from map (source of truth)
  auto block_it = param_map_.find(block);
  if (block_it != param_map_.end()) {
    block_it->second.erase(name);
  }

  // Remove from linked list (for output)
  InputBlock *pb = GetPtrToBlock(block);
  if (pb != nullptr) {
    InputLine *plast = pb->pline;
    for (InputLine *pl = pb->pline; pl != nullptr; pl = pl->pnext) {
      if (name.compare(pl->param_name) == 0) {
        // if head of list
        if (plast == pb->pline) {
          pb->pline = pl->pnext;
        } else {
          plast->pnext = pl->pnext;
        }
        delete pl;
        break;
      }
      plast = pl;
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
    auto *pvalue = FindParameter_(block, name);
    std::cout << std::endl
              << "Defaulting to <" << block << ">/" << name << " = "
              << ParamValueToString(*pvalue) << std::endl;
  }
}

void ParameterInput::CheckOrphans() const {
  std::set<std::pair<std::string, std::string>> orphans;
  for (InputBlock *pib = pfirst_block; pib != nullptr; pib = pib->pnext) {
    for (InputLine *pline = pib->pline; pline != nullptr; pline = pline->pnext) {
      auto key = std::make_pair(pib->block_name, pline->param_name);
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
//  \brief output entire InputBlock/InputLine hierarchy to specified stream

void ParameterInput::ParameterDump(std::ostream &os) {
  InputBlock *pb;
  InputLine *pl;
  std::string param_name, param_value;
  std::size_t len;

  os << "#------------------------- PAR_DUMP -------------------------" << std::endl;

  for (pb = pfirst_block; pb != nullptr; pb = pb->pnext) { // loop over InputBlocks
    os << "<" << pb->block_name << ">" << std::endl;       // write block name
    for (pl = pb->pline; pl != nullptr; pl = pl->pnext) {  // loop over InputLines
      param_name.assign(pl->param_name);
      param_value.assign(pl->param_value);

      len = pb->max_len_parname - param_name.length() + 1;
      param_name.append(len, ' '); // pad name to align vertically
      len = pb->max_len_parvalue - param_value.length() + 1;
      param_value.append(len, ' '); // pad value to align vertically

      os << param_name << "= " << param_value << pl->param_comment << std::endl;
    }
  }

  os << "#------------------------- PAR_DUMP -------------------------" << std::endl;
  os << "<par_end>" << std::endl; // finish with par-end (useful in restart files)
}

void ParameterInput::OutputParameterTable(std::ostream &os,
                                          const std::regex &block_regex) const {
  // Loop through once and store in a map for lexicographic ordering
  std::map<std::string, std::map<std::string, std::string>> csvblocks;
  for (InputBlock *pb = pfirst_block; pb != nullptr; pb = pb->pnext) {
    const std::string &block_name = pb->block_name;
    if (std::regex_match(block_name, block_regex)) {
      auto &csvlines = csvblocks[block_name];
      for (InputLine *pl = pb->pline; pl != nullptr; pl = pl->pnext) {
        const std::string &param_name = pl->param_name;
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
//! \fn InputLine* InputBlock::GetPtrToLine(std::string name)
//  \brief return pointer to InputLine containing specified parameter if it exists

InputLine *InputBlock::GetPtrToLine(std::string name) {
  for (InputLine *pl = pline; pl != nullptr; pl = pl->pnext) {
    if (name.compare(pl->param_name) == 0) return pl;
  }
  return nullptr;
}

//----------------------------------------------------------------------------------------
//! \fn std::string ParameterInput::ParamValueToString()
//  \brief Convert a ParamValue variant to string for linked list output

std::string ParameterInput::ParamValueToString(const ParamValue &value) {
  std::stringstream ss;

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
//! \fn void ParameterInput::AddParsedParameter()
//  \brief Generic interface for parsers to add parameters to storage
//  Can be called by any parser (text, Python, TOML, etc.) to populate param_map_

void ParameterInput::AddParsedParameter(const std::string &block, const std::string &name,
                                        const ParamValue &value,
                                        const std::string &comment) {
  PARTHENON_REQUIRE(!map_resolved_,
                    "Cannot add parameters after MarkResolved() has been called");

  // Track block insertion order (first appearance only)
  if (param_map_.find(block) == param_map_.end()) {
    block_order_.push_back(block);
  }

  // Add to param_map_ (source of truth)
  param_map_[block][name] = value;

  // Also update linked list for output compatibility
  auto *pb = FindOrAddBlock(block);
  AddParameter(pb, name, ParamValueToString(value), comment);
}

//----------------------------------------------------------------------------------------
//! \fn void ParameterInput::MarkResolved()
//  \brief Mark that all parsing is complete - no more parameters can be added

void ParameterInput::MarkResolved() { map_resolved_ = true; }

//----------------------------------------------------------------------------------------
//! \fn void ParameterInput::ResolveParametersToMap()
//  \brief Convert linked list structure to map for efficient access

void ParameterInput::ResolveParametersToMap() {
  // This function is now deprecated - the map is populated during parsing
  // via AddParsedParameter() calls from LoadFromStream() and ModifyFromCmdline().
  // Just mark as resolved.
  MarkResolved();
}

//----------------------------------------------------------------------------------------
//! \fn std::vector<std::string> ParameterInput::GetBlockNames()
//  \brief Return all block names in the input

std::vector<std::string> ParameterInput::GetBlockNames() const {
  // Use block_order_ to preserve insertion order
  return block_order_;
}

//----------------------------------------------------------------------------------------
//! \fn std::vector<std::string> ParameterInput::GetBlocksWithPrefix()
//  \brief Return all block names that start with the given prefix

std::vector<std::string>
ParameterInput::GetBlocksWithPrefix(const std::string &prefix) const {
  std::vector<std::string> matching_blocks;

  // Use block_order_ to preserve insertion order
  for (const auto &block_name : block_order_) {
    if (block_name.compare(0, prefix.length(), prefix) == 0) {
      matching_blocks.push_back(block_name);
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

  // Query the map for parameter names
  // NOTE: BREAKING CHANGE - Parameters are now returned in lexicographic order
  // (from std::map), not insertion order. Block order is preserved via block_order_,
  // but parameter order within blocks is not tracked.
  auto block_it = param_map_.find(block);
  if (block_it != param_map_.end()) {
    for (const auto &[param_name, param_value] : block_it->second) {
      param_names.push_back(param_name);
    }
  }

  return param_names;
}

//----------------------------------------------------------------------------------------
//! \fn ParamValue* ParameterInput::FindParameter_()
//  \brief Helper to find a parameter in the map, returns pointer for in-place
//  modification

ParameterInput::ParamValue *ParameterInput::FindParameter_(const std::string &block,
                                                           const std::string &name) {
  MarkResolved();
  auto block_it = param_map_.find(block);
  if (block_it != param_map_.end()) {
    auto param_it = block_it->second.find(name);
    if (param_it != block_it->second.end()) {
      return &(param_it->second);
    }
  }
  return nullptr;
}

//----------------------------------------------------------------------------------------
//! \fn template <typename T> std::optional<T> ParameterInput::GetFromMap_()
//  \brief Helper to get a typed parameter from map with caching
//  Returns nullopt if not in map, throws on type mismatch

template <typename T>
std::optional<T> ParameterInput::GetFromMap_(const std::string &block,
                                             const std::string &name) {
  MarkResolved();
  ParamValue *pvalue = FindParameter_(block, name);
  if (pvalue == nullptr) {
    return std::nullopt; // Not in map, caller should try linked list
  }

  // If it's an UnresolvedString, convert and cache
  if (std::holds_alternative<UnresolvedString>(*pvalue)) {
    T typed_val = ConvertParamValue<T>(*pvalue, block, name);
    *pvalue = typed_val; // Cache the typed value in the variant
    return typed_val;
  }

  // If it's already the correct type, return it
  if (std::holds_alternative<T>(*pvalue)) {
    return std::get<T>(*pvalue);
  }

  // Type mismatch - was previously resolved as a different type
  std::stringstream msg;
  msg << "### FATAL ERROR in ParameterInput::GetFromMap_" << std::endl
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
  variables.push_back(string_utils::trim(str));

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

// Explicit instantiations for GetFromMap_
template std::optional<int> ParameterInput::GetFromMap_<int>(const std::string &,
                                                             const std::string &);
template std::optional<Real> ParameterInput::GetFromMap_<Real>(const std::string &,
                                                               const std::string &);
template std::optional<bool> ParameterInput::GetFromMap_<bool>(const std::string &,
                                                               const std::string &);
template std::optional<std::string>
ParameterInput::GetFromMap_<std::string>(const std::string &, const std::string &);
template std::optional<std::vector<int>>
ParameterInput::GetFromMap_<std::vector<int>>(const std::string &, const std::string &);
template std::optional<std::vector<Real>>
ParameterInput::GetFromMap_<std::vector<Real>>(const std::string &, const std::string &);
template std::optional<std::vector<bool>>
ParameterInput::GetFromMap_<std::vector<bool>>(const std::string &, const std::string &);
template std::optional<std::vector<std::string>>
ParameterInput::GetFromMap_<std::vector<std::string>>(const std::string &,
                                                      const std::string &);

} // namespace parthenon
