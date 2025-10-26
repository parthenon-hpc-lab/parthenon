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

ParameterInput::ParameterInput(std::istringstream &is, const bool old, std::vector<std::string> input_filename, std::vector<std::string> mods)
    : pfirst_block{}, last_filename_{} {
  
  // restart file first
  if (old) {
    LoadFromOldRestart(is);
  } else {
    LoadFromRestart(is);
  }

  // then input files
  for ( auto &fname : input_filename) {
    IOWrapper infile;
    infile.Open(fname.c_str(), IOWrapper::FileMode::read);
    LoadFromFile(infile);
    infile.Close();
  }
  // now additional command line modifications
  if (!mods.empty()) {
    std::stringstream ss;
    for (auto &mod : mods) {
      ss << mod << "\n";
    }
    LoadFromStream(ss);
  }
  GenerateLinkedList();

}

ParameterInput::ParameterInput(std::vector<std::string> input_filename, std::vector<std::string> mods)
    : pfirst_block{}, last_filename_{} {
  // Do the input files first
  for ( auto &fname : input_filename) {
    IOWrapper infile;
    infile.Open(fname.c_str(), IOWrapper::FileMode::read);
    LoadFromFile(infile);
    infile.Close();
  }
  // now additional command line modifications
  if (!mods.empty()) {
    std::stringstream ss;
    for (auto &mod : mods) {
      ss << mod << "\n";
    }
    LoadFromStream(ss);
  }
  GenerateLinkedList();
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
 deck.Build(is);
}

void ParameterInput::LoadFromRestart(std::istringstream &is) {
  // When reading from a restart file, the string buffer has the <par_end> tag at the end.
  // This is not a real parameter block, so we need to remove it before passing it off.
  std::string sbuf = is.str();
  std::size_t loc = sbuf.find("<par_end>", 0);
  if (loc != std::string::npos) {
    sbuf = sbuf.substr(0, loc);
    std::istringstream ss = std::istringstream(sbuf);
    deck.Build(ss);
  } else {
    deck.Build(is);
  }
}

void ParameterInput::LoadFromOldRestart(std::istringstream &is) {
  // The old format did not enclose string parameters in quotes, so we need to
  // preprocess the input to add quotes around string values.
  std::stringstream ss;
  std::string line;
  while(std::getline(is, line)) {
    line.erase(std::remove_if(line.begin(), line.end(),
                              [](char c) { return std::isspace(c) && c != ' '; }),
               line.end());
    if (line.empty()) continue;                          // skip blank line
    std::size_t first_char = line.find_first_not_of(" ");
    if (line.compare(first_char, 1, "#") == 0) continue;      // skip comments
    if (line.compare(first_char, 9, "<par_end>") == 0) break; // stop on <par_end>

    std::size_t equal_pos = line.find('=');
    if (equal_pos != std::string::npos) {
      std::string name = line.substr(0, equal_pos);
      std::string value = line.substr(equal_pos + 1);
      name.erase(name.find_last_not_of(" \t\r\n") + 1);
      name.erase(0, name.find_first_not_of(" \t\r\n"));
      value.erase(value.find_last_not_of(" \t\r\n") + 1);
      value.erase(0, value.find_first_not_of(" \t\r\n"));
      bool is_numeric = !value.empty() && (std::all_of(value.begin(), value.end(),
                          [](char c){ return std::isdigit(c) || c == '.' || c == '-' || c == '+'; }));
      bool is_boolean = (value == "true" || value == "false" || value == "1" || value == "0");
      if (!is_numeric && !is_boolean) {
        // Add quotes around string values
        value = "\"" + value + "\"";
      }
      ss << name << " = " << value << "\n";
    } else {
      ss << line << "\n";
    }
  }
  deck.Build(ss);

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
    par.write(buf, ret); // add the buffer into the stjuream
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
    // remove <par_end>
    par = std::stringstream(sbuf.substr(0, loc)); 
  } while (ret == kBufSize); // till EOF (or par_end is found)

  // Now par contains the parameter inputs + some additional including <par_end>
  // Read the stream and load the parameters
  LoadFromStream(par);
  // Seek the file to the end of the header
  input.Seek(header);

  return;
}



void ParameterInput::GenerateLinkedList() {
  InputBlock *pib, *plast;
  plast = pfirst_block;
  pib = pfirst_block;

  for (auto &name : deck.GetSuitsInOrder()) {
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
    plast = pib;
  }
}


//----------------------------------------------------------------------------------------
//! \fn int ParameterInput::DoesParameterExist(const std::string & block, const
//! std::string & name)
//  \brief check whether parameter of given name in given block exists

int ParameterInput::DoesParameterExist(const std::string &block,
                                       const std::string &name) {
  return deck.DoesCardExist(block,name);
}

//----------------------------------------------------------------------------------------
//! \fn int ParameterInput::DoesBlockExist(const std::string & block)
//  \brief check whether block exists

int ParameterInput::DoesBlockExist(const std::string &block) {
  return deck.DoesSuitExist(block);
}

std::string ParameterInput::GetComment(const std::string &block,
                                       const std::string &name) {
  return deck.GetCard(block,name).GetComment();
}

//----------------------------------------------------------------------------------------
//! \fn int ParameterInput::GetInteger(const std::string & block, const std::string &
//! name)
//  \brief returns integer value of string stored in block/name

int ParameterInput::GetInteger(const std::string &block, const std::string &name,
                               const std::optional<std::string> &docstring) {

  auto val = deck.GetCardValue<int>(block,name);
  CheckAndUpdateQueries_<int>(block, name, docstring);

  return val;
}

//----------------------------------------------------------------------------------------
//! \fn Real ParameterInput::GetReal(const std::string & block, const std::string & name)
//  \brief returns real value of string stored in block/name

Real ParameterInput::GetReal(const std::string &block, const std::string &name,
                             const std::optional<std::string> &docstring) {
            
  std::stringstream msg;

  auto val = deck.GetCardValue<Real>(block,name);
  CheckAndUpdateQueries_<Real>(block, name, docstring);

  return val;
}

//----------------------------------------------------------------------------------------
//! \fn bool ParameterInput::GetBoolean(const std::string & block, const std::string &
//! name)
//  \brief returns boolean value of string stored in block/name

bool ParameterInput::GetBoolean(const std::string &block, const std::string &name,
                                const std::optional<std::string> &docstring) {
  auto val = deck.GetCardValue<bool>(block,name);
  CheckAndUpdateQueries_<bool>(block, name, docstring);

  return val;
}

//----------------------------------------------------------------------------------------
//! \fn std::string ParameterInput::GetString(const std::string & block, const std::string
//! & name)
//  \brief returns string stored in block/name

std::string ParameterInput::GetString(const std::string &block, const std::string &name,
                                      const std::optional<std::string> &docstring) {
  auto val = deck.GetCardValue<std::string>(block,name);
  CheckAndUpdateQueries_<std::string>(block, name, docstring);

  return val;
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
  CheckAndUpdateQueries_<int>(block, name, def_value, std::vector<int>{}, docstring);
  return deck.GetOrAddCardValue<int>(block,name,def_value);
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

  CheckAndUpdateQueries_<Real>(block, name, def_value, std::vector<Real>{}, docstring);
  return deck.GetOrAddCardValue<Real>(block,name,def_value);
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

  CheckAndUpdateQueries_<bool>(block, name, def_value, std::vector<bool>{}, docstring);
  return deck.GetOrAddCardValue<bool>(block,name,def_value);
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

  CheckAndUpdateQueries_<std::string>(block, name, def_value, std::vector<std::string>{},
                                      docstring);
  return deck.GetOrAddCardValue<std::string>(block,name,def_value);
}

std::string ParameterInput::GetOrAddString(const std::string &block,
                                           const std::string &name,
                                           const std::string &def_value,
                                           const std::vector<std::string> &allowed_values,
                                           const std::optional<std::string> &docstring) {
  auto val = deck.GetOrAddCardValue<std::string>(block,name,def_value);
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
  if (queries_.count(std::make_pair(block, name)) == 0) {
    CheckAndUpdateQueries_<int>(block, name, docstring);
  }

  deck.AddCard(block,name,value);

  UpdateQueryProvenance_(block, name, QueryRecord::OriginType::SetInCode);

  return value;
}

//----------------------------------------------------------------------------------------
//! \fn Real ParameterInput::SetReal(const std::string & block, const std::string &
//! name, Real value)
//  \brief updates a real parameter with full precision; creates it if it does not exist

Real ParameterInput::SetReal(const std::string &block, const std::string &name,
                             Real value, const std::optional<std::string> &docstring) {
  if (queries_.count(std::make_pair(block, name)) == 0) {
    CheckAndUpdateQueries_<Real>(block, name, docstring);
  }
  deck.AddCard(block,name,value);
  UpdateQueryProvenance_(block, name, QueryRecord::OriginType::SetInCode);

  return value;
}

//----------------------------------------------------------------------------------------
//! \fn bool ParameterInput::SetBoolean(const std::string & block, const std::string &
//! name, bool value)
//  \brief updates a boolean parameter; creates it if it does not exist

bool ParameterInput::SetBoolean(const std::string &block, const std::string &name,
                                bool value, const std::optional<std::string> &docstring) {
  if (queries_.count(std::make_pair(block, name)) == 0) {
    CheckAndUpdateQueries_<bool>(block, name, docstring);
  }

  deck.AddCard(block,name,value);

  UpdateQueryProvenance_(block, name, QueryRecord::OriginType::SetInCode);

  return value;
}

//----------------------------------------------------------------------------------------
//! \fn std::string ParameterInput::SetString(const std::string & block, const std::string
//! & name,
//                                            std::string  value)
//  \brief updates a string parameter; creates it if it does not exist

std::string ParameterInput::SetString(const std::string &block, const std::string &name,
                                      const std::string &value,
                                      const std::optional<std::string> &docstring) {
  if (queries_.count(std::make_pair(block, name)) == 0) {
    CheckAndUpdateQueries_<std::string>(block, name, docstring);
  }

  deck.AddCard(block,name,value);

  UpdateQueryProvenance_(block, name, QueryRecord::OriginType::SetInCode);

  return value;
}

void ParameterInput::RemoveParameter(const std::string &block, const std::string &name) {
  deck.RemoveCard(block,name);
  auto key = std::make_pair(block, name);
  auto it = queries_.find(key);
  if (it != queries_.end()) {
    queries_.erase(it);
  }
}

void ParameterInput::CheckRequired(const std::string &block, const std::string &name) {
  bool missing = true;
  if (DoesParameterExist(block, name)) {
    missing = (GetComment(block, name) == "Default value added at run time");
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
    defaulted = (GetComment(block, name) == "Default value added at run time");
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

  deck.WriteDeck(os);

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

} // namespace parthenon
