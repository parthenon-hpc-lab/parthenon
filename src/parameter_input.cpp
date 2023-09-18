//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020-2023. Triad National Security, LLC. All rights reserved.
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
  std::string line, block_name, param_name, param_value, param_comment;
  std::size_t first_char, last_char;
  std::stringstream msg;
  InputBlock *pib{};
  int line_num{-1}, blocks_found{0};

  std::string multiline_name, multiline_value, multiline_comment;
  bool continuing = false;

  while (is.good()) {
    std::getline(is, line);
    line_num++;
    if (line.find('\t') != std::string::npos) {
      line.erase(std::remove(line.begin(), line.end(), '\t'), line.end());
      // msg << "### FATAL ERROR in function [ParameterInput::LoadFromStream]"
      //     << std::endl << "Tab characters are forbidden in input files";
      // PARTHENON_FAIL(msg);
    }
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
    }           // end "a new block was found"

    // if line does not contain a block name or skippable information (comments,
    // whitespace), it must contain a parameter value
    if (blocks_found == 0) {
      msg << "### FATAL ERROR in function [ParameterInput::LoadFromStream]" << std::endl
          << "Input file must specify a block name before the first"
          << " parameter = value line";
      PARTHENON_FAIL(msg);
    }
    // parse line and add name/value/comment strings (if found) to current block name
    bool continuation = ParseLine(pib, line, param_name, param_value, param_comment);
    if (continuing || continuation) {
      multiline_name += param_name;
      multiline_value += param_value;
      multiline_comment += param_comment;
      continuing = true;
    }

    if (continuing && !continuation) {
      continuing = false;
      param_name = multiline_name;
      param_value = multiline_value;
      param_comment = multiline_comment;
      // Clear the multiline_x buffers to hold the next multi-line value
      multiline_name = "";
      multiline_value = "";
      multiline_comment = "";
    }

    if (!continuing) {
      if (param_name != "") {
        AddParameter(pib, param_name, param_value, param_comment);
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
    line.erase(0, len + 1);
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
  std::string input_text, block, name, value;
  std::stringstream msg;
  InputBlock *pb;
  InputLine *pl;

  for (int i = 1; i < argc; i++) {
    input_text = argv[i];
    std::size_t equal_posn = input_text.find_first_of("=");     // first "=" character
    std::size_t slash_posn = input_text.rfind("/", equal_posn); // last "/" before "="

    if (slash_posn > equal_posn) {
      msg << "'/' used as value (rhs of =) when modifying " << input_text << "."
          << " Please update value of change "
          << "logic in ModifyFromCmdline function.";
      PARTHENON_FAIL(msg.str().c_str());
    }

    // skip if either "/" or "=" do not exist in input
    if ((slash_posn == std::string::npos) || (equal_posn == std::string::npos)) continue;

    // extract block/name/value strings
    block = input_text.substr(0, slash_posn);
    name = input_text.substr(slash_posn + 1, (equal_posn - slash_posn - 1));
    value = input_text.substr(equal_posn + 1, std::string::npos);

    // get pointer to node with same block name in singly linked list of InputBlocks
    pb = GetPtrToBlock(block);
    if (pb == nullptr) {
      if (Globals::my_rank == 0) {
        msg << "In function [ParameterInput::ModifyFromCmdline]:" << std::endl
            << "               Block name '" << block
            << "' on command line not found in input/restart file. Block will be added.";
        PARTHENON_WARN(msg);
      }
      pb = FindOrAddBlock(block);
    }

    // get pointer to node with same parameter name in singly linked list of InputLines
    pl = pb->GetPtrToLine(name);
    if (pl == nullptr) {
      if (Globals::my_rank == 0) {
        msg << "In function [ParameterInput::ModifyFromCmdline]:" << std::endl
            << "               Parameter '" << name << "' in block '" << block
            << "' on command line not found in input/restart file. Parameter will be "
               "added.";
        PARTHENON_WARN(msg);
      }
      AddParameter(pb, name, value, " # Added from command line");

    } else {
      pl->param_value.assign(value); // replace existing value
    }

    if (value.length() > pb->max_len_parvalue) pb->max_len_parvalue = value.length();
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
  InputLine *pl;
  InputBlock *pb;
  pb = GetPtrToBlock(block);
  if (pb == nullptr) return 0;
  pl = pb->GetPtrToLine(name);
  return (pl == nullptr ? 0 : 1);
}

//----------------------------------------------------------------------------------------
//! \fn int ParameterInput::DoesBlockExist(const std::string & block)
//  \brief check whether block exists

int ParameterInput::DoesBlockExist(const std::string &block) {
  InputBlock *pb = GetPtrToBlock(block);
  if (pb == nullptr) return 0;
  return 1;
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

int ParameterInput::GetInteger(const std::string &block, const std::string &name) {
  InputBlock *pb;
  InputLine *pl;
  std::stringstream msg;

  // get pointer to node with same block name in singly linked list of InputBlocks
  pb = GetPtrToBlock(block);
  if (pb == nullptr) {
    msg << "### FATAL ERROR in function [ParameterInput::GetInteger]" << std::endl
        << "Block name '" << block << "' not found when trying to set value "
        << "for parameter '" << name << "'";
    PARTHENON_FAIL(msg);
  }

  // get pointer to node with same parameter name in singly linked list of InputLines
  pl = pb->GetPtrToLine(name);
  if (pl == nullptr) {
    msg << "### FATAL ERROR in function [ParameterInput::GetInteger]" << std::endl
        << "Parameter name '" << name << "' not found in block '" << block << "'";
    PARTHENON_FAIL(msg);
  }

  std::string val = pl->param_value;

  // Convert string to integer and return value
  return stoi(val);
}

//----------------------------------------------------------------------------------------
//! \fn Real ParameterInput::GetReal(const std::string & block, const std::string & name)
//  \brief returns real value of string stored in block/name

Real ParameterInput::GetReal(const std::string &block, const std::string &name) {
  InputBlock *pb;
  InputLine *pl;
  std::stringstream msg;

  // get pointer to node with same block name in singly linked list of InputBlocks
  pb = GetPtrToBlock(block);
  if (pb == nullptr) {
    msg << "### FATAL ERROR in function [ParameterInput::GetReal]" << std::endl
        << "Block name '" << block << "' not found when trying to set value "
        << "for parameter '" << name << "'";
    PARTHENON_FAIL(msg);
  }

  // get pointer to node with same parameter name in singly linked list of InputLines
  pl = pb->GetPtrToLine(name);
  if (pl == nullptr) {
    msg << "### FATAL ERROR in function [ParameterInput::GetReal]" << std::endl
        << "Parameter name '" << name << "' not found in block '" << block << "'";
    PARTHENON_FAIL(msg);
  }

  std::string val = pl->param_value;

  // Convert string to real and return value
  return static_cast<Real>(atof(val.c_str()));
}

//----------------------------------------------------------------------------------------
//! \fn bool ParameterInput::GetBoolean(const std::string & block, const std::string &
//! name)
//  \brief returns boolean value of string stored in block/name

bool ParameterInput::GetBoolean(const std::string &block, const std::string &name) {
  InputBlock *pb;
  InputLine *pl;
  std::stringstream msg;

  // get pointer to node with same block name in singly linked list of InputBlocks
  pb = GetPtrToBlock(block);
  if (pb == nullptr) {
    msg << "### FATAL ERROR in function [ParameterInput::GetBoolean]" << std::endl
        << "Block name '" << block << "' not found when trying to set value "
        << "for parameter '" << name << "'";
    PARTHENON_FAIL(msg);
  }

  // get pointer to node with same parameter name in singly linked list of InputLines
  pl = pb->GetPtrToLine(name);
  if (pl == nullptr) {
    msg << "### FATAL ERROR in function [ParameterInput::GetBoolean]" << std::endl
        << "Parameter name '" << name << "' not found in block '" << block << "'";
    PARTHENON_FAIL(msg);
  }

  std::string val = pl->param_value;
  return stob(val);
}

//----------------------------------------------------------------------------------------
//! \fn std::string ParameterInput::GetString(const std::string & block, const std::string
//! & name)
//  \brief returns string stored in block/name

std::string ParameterInput::GetString(const std::string &block, const std::string &name) {
  InputBlock *pb;
  InputLine *pl;
  std::stringstream msg;

  // get pointer to node with same block name in singly linked list of InputBlocks
  pb = GetPtrToBlock(block);
  if (pb == nullptr) {
    msg << "### FATAL ERROR in function [ParameterInput::GetString]" << std::endl
        << "Block name '" << block << "' not found when trying to set value "
        << "for parameter '" << name << "'";
    PARTHENON_FAIL(msg);
  }

  // get pointer to node with same parameter name in singly linked list of InputLines
  pl = pb->GetPtrToLine(name);
  if (pl == nullptr) {
    msg << "### FATAL ERROR in function [ParameterInput::GetString]" << std::endl
        << "Parameter name '" << name << "' not found in block '" << block << "'";
    PARTHENON_FAIL(msg);
  }

  std::string val = pl->param_value;

  // return value
  return val;
}

std::string ParameterInput::GetString(const std::string &block, const std::string &name,
                                      const std::vector<std::string> &allowed_values) {
  auto val = GetString(block, name);
  CheckAllowedValues_(block, name, val, allowed_values);
  return val;
}

//----------------------------------------------------------------------------------------
//! \fn int ParameterInput::GetOrAddInteger(const std::string & block, const std::string &
//! name,
//    int default_value)
//  \brief returns integer value stored in block/name if it exists, or creates and sets
//  value to def_value if it does not exist

int ParameterInput::GetOrAddInteger(const std::string &block, const std::string &name,
                                    int def_value) {
  InputBlock *pb;
  InputLine *pl;
  std::stringstream ss_value;
  int ret;

  if (DoesParameterExist(block, name)) {
    pb = GetPtrToBlock(block);
    pl = pb->GetPtrToLine(name);
    std::string val = pl->param_value;
    ret = stoi(val);
  } else {
    pb = FindOrAddBlock(block);
    ss_value << def_value;
    AddParameter(pb, name, ss_value.str(), "# Default value added at run time");
    ret = def_value;
  }
  return ret;
}

//----------------------------------------------------------------------------------------
//! \fn Real ParameterInput::GetOrAddReal(const std::string & block, const std::string &
//! name,
//    Real def_value)
//  \brief returns real value stored in block/name if it exists, or creates and sets
//  value to def_value if it does not exist

Real ParameterInput::GetOrAddReal(const std::string &block, const std::string &name,
                                  Real def_value) {
  InputBlock *pb;
  InputLine *pl;
  std::stringstream ss_value;
  Real ret;

  if (DoesParameterExist(block, name)) {
    pb = GetPtrToBlock(block);
    pl = pb->GetPtrToLine(name);
    std::string val = pl->param_value;
    ret = static_cast<Real>(atof(val.c_str()));
  } else {
    pb = FindOrAddBlock(block);
    static_assert(sizeof(Real) <= sizeof(double), "Real is greater than double!");
    ss_value.precision(std::numeric_limits<double>::max_digits10);
    ss_value << def_value;
    AddParameter(pb, name, ss_value.str(), "# Default value added at run time");
    ret = def_value;
  }
  return ret;
}

//----------------------------------------------------------------------------------------
//! \fn bool ParameterInput::GetOrAddBoolean(const std::string & block, const std::string
//! & name,
//    bool def_value)
//  \brief returns boolean value stored in block/name if it exists, or creates and sets
//  value to def_value if it does not exist

bool ParameterInput::GetOrAddBoolean(const std::string &block, const std::string &name,
                                     bool def_value) {
  InputBlock *pb;
  InputLine *pl;
  std::stringstream ss_value;
  bool ret;

  if (DoesParameterExist(block, name)) {
    pb = GetPtrToBlock(block);
    pl = pb->GetPtrToLine(name);
    std::string val = pl->param_value;
    if (val.compare(0, 1, "0") == 0 || val.compare(0, 1, "1") == 0) {
      ret = static_cast<bool>(stoi(val));
    } else {
      std::transform(val.begin(), val.end(), val.begin(), ::tolower);
      std::istringstream is(val);
      is >> std::boolalpha >> ret;
    }
  } else {
    pb = FindOrAddBlock(block);
    ss_value << def_value;
    AddParameter(pb, name, ss_value.str(), "# Default value added at run time");
    ret = def_value;
  }
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
                                           const std::string &def_value) {
  InputBlock *pb;
  InputLine *pl;
  std::stringstream ss_value;
  std::string ret;

  if (DoesParameterExist(block, name)) {
    pb = GetPtrToBlock(block);
    pl = pb->GetPtrToLine(name);
    ret = pl->param_value;
  } else {
    pb = FindOrAddBlock(block);
    AddParameter(pb, name, def_value, "# Default value added at run time");
    ret = def_value;
  }
  return ret;
}
std::string
ParameterInput::GetOrAddString(const std::string &block, const std::string &name,
                               const std::string &def_value,
                               const std::vector<std::string> &allowed_values) {
  auto val = GetOrAddString(block, name, def_value);
  CheckAllowedValues_(block, name, val, allowed_values);
  return val;
}

//----------------------------------------------------------------------------------------
//! \fn int ParameterInput::SetInteger(const std::string & block, const std::string &
//! name, int value)
//  \brief updates an integer parameter; creates it if it does not exist

int ParameterInput::SetInteger(const std::string &block, const std::string &name,
                               int value) {
  InputBlock *pb;
  std::stringstream ss_value;

  pb = FindOrAddBlock(block);
  ss_value << value;
  AddParameter(pb, name, ss_value.str(), "# Updated during run time");
  return value;
}

//----------------------------------------------------------------------------------------
//! \fn Real ParameterInput::SetReal(const std::string & block, const std::string &
//! name, Real value)
//  \brief updates a real parameter with full precision; creates it if it does not exist

Real ParameterInput::SetReal(const std::string &block, const std::string &name,
                             Real value) {
  InputBlock *pb;
  std::stringstream ss_value;

  pb = FindOrAddBlock(block);
  static_assert(sizeof(Real) <= sizeof(double), "Real is greater than double!");
  ss_value.precision(std::numeric_limits<double>::max_digits10);
  ss_value << value;
  AddParameter(pb, name, ss_value.str(), "# Updated during run time");
  return value;
}

//----------------------------------------------------------------------------------------
//! \fn bool ParameterInput::SetBoolean(const std::string & block, const std::string &
//! name, bool value)
//  \brief updates a boolean parameter; creates it if it does not exist

bool ParameterInput::SetBoolean(const std::string &block, const std::string &name,
                                bool value) {
  InputBlock *pb;
  std::stringstream ss_value;

  pb = FindOrAddBlock(block);
  ss_value << value;
  AddParameter(pb, name, ss_value.str(), "# Updated during run time");
  return value;
}

//----------------------------------------------------------------------------------------
//! \fn std::string ParameterInput::SetString(const std::string & block, const std::string
//! & name,
//                                            std::string  value)
//  \brief updates a string parameter; creates it if it does not exist

std::string ParameterInput::SetString(const std::string &block, const std::string &name,
                                      const std::string &value) {
  InputBlock *pb;

  pb = FindOrAddBlock(block);
  AddParameter(pb, name, value, "# Updated during run time");
  return value;
}

//----------------------------------------------------------------------------------------
//! \fn void ParameterInput::RollbackNextTime()
//  \brief rollback next_time by dt for each output block

void ParameterInput::RollbackNextTime() {
  InputBlock *pb = pfirst_block;
  InputLine *pl;
  std::stringstream msg;
  Real next_time;

  while (pb != nullptr) {
    if (pb->block_name.compare(0, 16, "parthenon/output") == 0) {
      pl = pb->GetPtrToLine("next_time");
      if (pl == nullptr) {
        msg << "### FATAL ERROR in function [ParameterInput::RollbackNextTime]"
            << std::endl
            << "Parameter name 'next_time' not found in block '" << pb->block_name << "'";
        PARTHENON_FAIL(msg);
      }
      next_time = static_cast<Real>(atof(pl->param_value.c_str()));
      pl = pb->GetPtrToLine("dt");
      if (pl == nullptr) {
        msg << "### FATAL ERROR in function [ParameterInput::RollbackNextTime]"
            << std::endl
            << "Parameter name 'dt' not found in block '" << pb->block_name << "'";
        PARTHENON_FAIL(msg);
      }
      next_time -= static_cast<Real>(atof(pl->param_value.c_str()));
      msg << next_time;
      // AddParameter(pb, "next_time", msg.str().c_str(), "# Updated during run time");
      SetReal(pb->block_name, "next_time", next_time);
    }
    pb = pb->pnext;
  }
}

//----------------------------------------------------------------------------------------
//! \fn void ParameterInput::ForwardNextTime()
//  \brief add dt to next_time until next_time >  mesh_time - dt for each output block

void ParameterInput::ForwardNextTime(Real mesh_time) {
  InputBlock *pb = pfirst_block;
  InputLine *pl;
  Real next_time;
  Real dt0, dt;
  bool fresh = false;

  while (pb != nullptr) {
    if (pb->block_name.compare(0, 16, "parthenon/output") == 0) {
      std::stringstream msg;
      pl = pb->GetPtrToLine("next_time");
      if (pl == nullptr) {
        next_time = mesh_time;
        // This is a freshly added output
        fresh = true;
      } else {
        next_time = static_cast<Real>(atof(pl->param_value.c_str()));
      }
      pl = pb->GetPtrToLine("dt");
      if (pl == nullptr) {
        msg << "### FATAL ERROR in function [ParameterInput::ForwardNextTime]"
            << std::endl
            << "Parameter name 'dt' not found in block '" << pb->block_name << "'";
        PARTHENON_FAIL(msg);
      }
      dt0 = static_cast<Real>(atof(pl->param_value.c_str()));
      dt = dt0 * static_cast<int>((mesh_time - next_time) / dt0) + dt0;
      if (dt > 0) {
        next_time += dt;
        // If the user has added a new/fresh output round to multiple of dt0,
        // and make sure that mesh_time - dt0 < next_time < mesh_time,
        // to ensure immediate writing
        if (fresh) next_time -= std::fmod(next_time, dt0) + dt0;
      }
      msg << next_time;
      AddParameter(pb, "next_time", msg.str(), "# Updated during run time");
    }
    pb = pb->pnext;
  }
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
    std::cout << std::endl
              << "Defaulting to <" << block << ">/" << name << " = "
              << GetString(block, name) << std::endl;
  }
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

//----------------------------------------------------------------------------------------
//! \fn InputLine* InputBlock::GetPtrToLine(std::string name)
//  \brief return pointer to InputLine containing specified parameter if it exists

InputLine *InputBlock::GetPtrToLine(std::string name) {
  for (InputLine *pl = pline; pl != nullptr; pl = pl->pnext) {
    if (name.compare(pl->param_name) == 0) return pl;
  }
  return nullptr;
}

} // namespace parthenon
