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

#include <toml.hpp>

#include "globals.hpp"
#include "inputs/parameter_input.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {

std::string ParameterPath(const std::string block, const std::string name) {
  if (!std::count(block.begin(), block.end(), '/')) {
    if (name == "") {
      return block;
    } else {
      return block + "." + name;
    }
  } else {
    std::string b(block);
    std::replace(b.begin(), b.end(), '/', '.');
    if (name == "") {
      return b;
    } else {
      return b + "." + name;
    }
  }
}

void ParameterInput::Merge(toml::table &a, const toml::table &b, bool check_dups) {
  b.for_each([&](const toml::key &key, auto &&el) {
    recursive_merge(a, b, key, el, check_dups);
  });
}

void ParameterInput::recursive_get_paths(const toml::table &a, toml::path prefix,
                                         const toml::key &key,
                                         std::vector<toml::path> &paths) const {
  if (a[key].is<toml::table>()) {
    const toml::table &achild = a[key].ref<toml::table>();
    toml::path block = (prefix != toml::path("")) ? toml::path(prefix.append(key.str()))
                                                  : toml::path(key.str());
    achild.for_each([&](const toml::key &key, auto &&el) {
      recursive_get_paths(achild, block, key, paths);
    });
  } else {
    paths.push_back(toml::path(prefix.append(key.str())));
  }
}
std::vector<std::string> ParameterInput::GetAllPaths(const toml::table &a) const {
  std::vector<toml::path> paths;
  a.for_each([&](const toml::key &key, auto &&el) {
    recursive_get_paths(a, toml::path(""), key, paths);
  });
  // Could have a version that returns these path objs,
  // but probably everyone wants strings?
  std::vector<std::string> path_strings;
  for (auto path : paths) {
    path_strings.push_back(path.str());
  }
  return path_strings;
}

toml::table ParameterInput::Blocks() { return parameters_; }
toml::table ParameterInput::Blocks(const char *path) {
  return GetPath<toml::table>(path);
}
toml::table ParameterInput::Blocks(std::string &path) {
  return GetPath<toml::table>(path);
}
// Get const copies of entire contents, primarily for hashing
const toml::table ParameterInput::GetAll() const { return parameters_; }

RecordOrigin ParameterInput::GetOrigin(const std::string &path) {
  return queries_.at(path).origin;
}

void ParameterInput::LoadFile(const std::string fname, bool check_for_overrides) {
  std::stringstream contents;
  char *buf = reinterpret_cast<char *>(calloc(sizeof(char), max_input_filesize_));

  IOWrapper infile;
  infile.Open(fname.c_str(), IOWrapper::FileMode::read);
  infile.Read_all(buf, sizeof(char), max_input_filesize_);
  infile.Close();
  contents.write(buf, max_input_filesize_);
  free(buf);

  LoadFromStream(contents, RecordOrigin(fname), check_for_overrides);

  return;
}

void ParameterInput::LoadFromStream(std::istream &is, const RecordOrigin &origin,
                                    bool check_for_overrides) {
  std::stringstream ss;
  ss << is.rdbuf();
  std::string input = ss.str();

  // Stash this for archival use
  PARTHENON_REQUIRE_THROWS(pre_parsed_inputs_.count(origin) == 0,
                           "Each input must only be processed once");
  pre_parsed_inputs_[origin] = input;

  // Remove all null bytes: it's faster & toml doesn't like them
  input.erase(std::remove(input.begin(), input.end(), '\00'), input.end());
  // If n(<) > n([), we're parsing an old-style file. Otherwise, TOML
  int nangle = std::count(input.begin(), input.end(), '<');
  int nsquare = std::count(input.begin(), input.end(), '[');
  auto new_parameters = toml::table();
  if (nangle > nsquare) {
    is.seekg(is.beg);
    new_parameters = LegacyParse(is, origin);
  } else {
    new_parameters = toml::parse(input);
  }
  // Merge from different inputs, only check for overrides if asked
  Merge(parameters_, new_parameters, check_for_overrides);
}

void ParameterInput::ModifyFromCmdline(int argc, char *argv[]) {
  std::string input_text, path, value;
  std::stringstream msg, cli_record_string;

  for (int i = 1; i < argc; i++) {
    input_text = argv[i];

    std::size_t equal_posn = input_text.find_first_of("="); // first "=" character

    // Only parse arguments with '='
    if (equal_posn == std::string::npos) continue;

    // stash the argument for archival purposes
    cli_record_string << input_text << std::endl;

    // Sanitize block + name together, but only if there are no dots
    path = input_text.substr(0, equal_posn);
    // This replaces all '/', so they can't be used in varnames which will be passed
    // on the command line, even when using pure TOML inputs.
    // ('/' obviously can't be used in block names anyway if we want back-compat)
    // Alternatively we could check for '.', which would forbid '.' in old-style
    // blocks and varnames retroactively.
    path = ParameterPath(path, "");
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
    AddParameter_(parameters_, path, value, RecordOrigin(OriginType::CommandLine));
  }
  pre_parsed_inputs_[RecordOrigin(OriginType::CommandLine)] = cli_record_string.str();
}

int ParameterInput::DoesParameterExist(const std::string &block,
                                       const std::string &name) {
  return DoesParameterExist(ParameterPath(block, name));
}
int ParameterInput::DoesParameterExist(const std::string &path) {
  return !!parameters_.at_path(path);
}
int ParameterInput::DoesBlockExist(const std::string &block) {
  return parameters_.contains(ParameterPath(block, ""));
}

void ParameterInput::CheckRequired(const std::string &block, const std::string &name) {
  return CheckRequired(ParameterPath(block, name));
}
void ParameterInput::CheckRequired(const std::string &path) {
  bool exists = DoesParameterExist(path) && (GetOrigin(path).type != OriginType::Default);
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
  return CheckDesired(ParameterPath(block, name));
}
void ParameterInput::CheckDesired(const std::string &path) {
  bool missing = true;
  bool defaulted = false;
  if (DoesParameterExist(path)) {
    missing = false;
    defaulted = (GetOrigin(path).type == OriginType::Default);
  }
  if (missing) {
    std::cout << std::endl
              << "### WARNING in CheckDesired:" << std::endl
              << "Parameter file missing desired field " << path << std::endl;
  }
  if (defaulted) { // Could look up the default here, but it's set to that anyway
    std::cout << std::endl
              << "Defaulting to " << path << " = " << parameters_.at_path(path)
              << std::endl;
  }
}

void ParameterInput::CheckOrphans() const {
  std::size_t count = 0;
  std::stringstream msg;
  msg << "The following input parameters are set but unused:\n";
  for (auto path : GetAllPaths(parameters_)) {
    auto &query = queries_.at(path);
    if (!query.requested) {
      msg << path << ", with " << query.origin << "\n";
      count++;
    }
  }
  msg << std::endl;
  if ((Globals::my_rank == 0) && (count > 0)) {
    PARTHENON_WARN(msg);
  }
}

void ParameterInput::ParameterDump(std::ostream &os) { os << parameters_ << "\n"; }

toml::table ParameterInput::LegacyParse(std::istream &is, const RecordOrigin &origin) {
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
        PARTHENON_THROW(msg);
      }
      first_char++;
      last_char = (line.find_first_of(">", first_char));
      block_name.assign(line, first_char, last_char - 1); // extract block name
      block_name = ParameterPath(block_name, "");

      if (last_char == std::string::npos) {
        msg << "### FATAL ERROR in function [ParameterInput::LegacyParse]" << std::endl
            << "Block name '" << block_name << "' in the input stream'"
            << "' not properly ended";
        PARTHENON_THROW(msg);
      }

      blocks_found++;
      continue; // skip to next line if block name was found
    }           // end "a new block was found"

    // if line does not contain a block name or skippable information (comments,
    // whitespace), it must contain a parameter value
    if (blocks_found == 0) {
      msg << "### FATAL ERROR in function [ParameterInput::LegacyParse]" << std::endl
          << "Input file must specify a block name before the first"
          << " parameter = value line";
      PARTHENON_THROW(msg);
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
        AddParameter_(tmp_tbl, ParameterPath(block_name, param_name), param_value, origin,
                      true);
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

void ParameterInput::OutputParameterTable(std::ostream &os,
                                          const std::regex &block_regex) const {
  // TOML's node-types are a bit verbose. This is simpler.
  auto SimpleName = [](const toml::node_type tp) -> std::string {
    if (tp == toml::node_type::integer) return "int";
    if (tp == toml::node_type::boolean) return "bool";
    if (tp == toml::node_type::floating_point) return "Real";
    std::stringstream ss;
    ss << tp;
    return ss.str();
  };
  // Loop through parameters.  Already alphabetical, just gotta split block/name
  os << "block,parameters,type,default,description" << std::endl;
  std::string last_block_name = "";
  for (auto path : GetAllPaths(parameters_)) {
    // Yeah, GetAllPaths returns strings.  Make it back into a path
    auto toml_path = toml::path(path);
    std::string block_name, param_name;
    if (toml_path.size() < 2) {
      param_name = path;
      block_name = "root";
    } else {
      std::size_t idx = toml_path.size() - 1;
      param_name = toml_path[idx].key();
      block_name = toml_path.subpath(0, idx).str();
    }
    // Filter on block name fitting user regex
    if (std::regex_match(block_name, block_regex)) {
      // Output blank lines on block change
      if (block_name != last_block_name && last_block_name != "")
        os << "\"\",\"\",\"\",\"\",\"\"" << std::endl;
      last_block_name = block_name;
      /* clang-format off */
      if (queries_.count(path) > 0) {
        auto record = queries_.at(path);
        std::stringstream ss;
        ss << "\"" << block_name << "\""
           << "," << "\"" << param_name << "\""
           << "," << "\"" << SimpleName(GetTypePath(path)) << "\""
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
        os << ss.str() << std::endl;
      }
    }
  }
}

bool operator<(const RecordOrigin &lhs, const RecordOrigin &rhs) {
  return (lhs.type < rhs.type) || (lhs.type == rhs.type && lhs.file < rhs.file);
}
std::ostream &operator<<(std::ostream &os, RecordOrigin::Type type) {
  switch (type) {
  case RecordOrigin::Type::None:
    return os << "None";
  case RecordOrigin::Type::InputFile:
    return os << "InputFile";
  case RecordOrigin::Type::Restart:
    return os << "Restart";
  case RecordOrigin::Type::Default:
    return os << "Default";
  case RecordOrigin::Type::SetInCode:
    return os << "SetInCode";
  case RecordOrigin::Type::CommandLine:
    return os << "CommandLine";
  default:
    return os << "Unknown";
  }
}
std::ostream &operator<<(std::ostream &os, const RecordOrigin &origin) {
  os << "Origin Type: " << origin.type;
  if (origin.HasFile()) {
    os << ", File: \"" << origin.file << "\"";
  }
  return os;
}
std::string RecordOrigin::ToString() const {
  std::stringstream ss;
  ss << *this;
  return ss.str();
}

} // namespace parthenon
