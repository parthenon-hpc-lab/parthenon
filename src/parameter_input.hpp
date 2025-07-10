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
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================

#ifndef PARAMETER_INPUT_HPP_
#define PARAMETER_INPUT_HPP_
//! \file parameter_input.hpp
//  \brief definition of class ParameterInput
// Contains data structures used to store, and functions used to access, parameters
// read from the input file.

#include <algorithm>
#include <any>
#include <cstddef>
#include <map>
#include <optional>
#include <ostream>
#include <regex>
#include <set>
#include <string>
#include <typeinfo>
#include <utility> // for std::forward, std::pair
#include <vector>

#include <toml.hpp>

#include "config.hpp"
#include "defs.hpp"
#include "globals.hpp"
#include "outputs/io_wrapper.hpp"
#include "utils/hash.hpp"
#include "utils/sort.hpp"
#include "utils/string_utils.hpp"
#include "utils/utils.hpp"

namespace parthenon {

// We need to overload the stream operator for containers to output
// something sensible
// TODO(JMM): I'm pretty sure this is incredibly dangerous, even in the
// parthenon namespace. Once we're on TOML, don't do this. Convert the
// vector to a toml::array which already has an overloaded ostream.
// Alternatively, we could try something insane like a printable
// vector that automatically casts to our container type.
/*
template <typename T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &container) {
  std::size_t i = 0;
  os << "[";
  for (const T &elem : container) {
    os << elem;
    if (i < container.size() - 1) {
      os << ", ";
    }
  }
  os << "]";
  return os;
}
*/

std::string ParameterPath(const std::string block, const std::string name);

struct QueryRecord {
  enum class OriginType { None, Input, Restart, Default, SetInCode, CommandLine };
  OriginType origin_type = OriginType::None; // This should never persist
  std::string origin_file;
  std::any default_value; // std::any::has_value to check if default
                          // val exists
  std::string
      default_value_str; // used for output, so we don't have to mess with types later
  std::vector<std::any> allowed_values;      // size to check if allowed values exist
  std::vector<std::string> allowed_vals_str; // used for output
  std::optional<std::string> docstring; // std::optional::has_value to check if exists
  template <typename T>
  static std::string ToString(const T &val) {
    std::stringstream ss;
    if constexpr (std::is_same_v<T, double>) {
      ss.precision(8); // max digits is totally unreadable
      // ss.precision(std::numeric_limits<T>::max_digits10);
    }
    ss << val;
    return ss.str();
  }
};

// This can be used to tell the params infrastructure that the default
// value of one parameter depends on another one
class ParameterRef {
 public:
  ParameterRef(const std::string &block, const std::string &name)
      : path_(ParameterPath(block, name)) {}
  explicit ParameterRef(const std::string &path) : path_(path) {}
  std::string CanonicalPath() const { return path_; }

 private:
  const std::string path_;
};

class ParameterInput {
  friend class std::hash<ParameterInput>;
  using OriginType = QueryRecord::OriginType;

 public:
  // constructor/destructor
  ParameterInput() { parameters_ = toml::table(); }
  explicit ParameterInput(std::string input_filename) {
    parameters_ = toml::table();
    LoadFile(input_filename);
  }
  ~ParameterInput() {}

  // functions
  void LoadFromStream(std::istream &is, std::string fname = "",
                      bool check_for_overrides = false);
  void LoadFile(std::string fname, bool check_for_overrides = false);
  void ModifyFromCmdline(int argc, char *argv[]);

  void ParameterDump(std::ostream &os);

  int DoesParameterExist(const std::string &block, const std::string &name);
  int DoesParameterExist(const std::string &path);
  int DoesBlockExist(const std::string &block);

  // TODO(JMM): Make this more general?
  void OutputParameterTable(std::ostream &os,
                            const std::regex &block_regex = std::regex("(.*)")) const;

  void CheckRequired(const std::string &block, const std::string &name);
  void CheckRequired(const std::string &path);
  void CheckDesired(const std::string &block, const std::string &name);
  void CheckDesired(const std::string &path);
  void CheckOrphans() const;

  toml::table Blocks();
  toml::table Blocks(const char *path);
  toml::table Blocks(std::string &path);
  const toml::table GetAll() const;
  std::vector<std::string> GetAllPaths(const toml::table &a) const;

  OriginType GetOrigin(const std::string &path);

  // toml++ only supports int64_t
  template <typename... Args>
  int GetInteger(Args &&...args) {
    return static_cast<int>(Get<int64_t>(std::forward<Args>(args)...));
  }
  template <typename... Args>
  int GetOrAddInteger(Args &&...args) {
    return static_cast<int>(GetOrAdd<int64_t>(std::forward<Args>(args)...));
  }
  template <typename... Args>
  int SetInteger(Args &&...args) {
    return static_cast<int>(Set<int64_t>(std::forward<Args>(args)...));
  }
  // toml++ only supports double
  template <typename... Args>
  double GetReal(Args &&...args) {
    return Get<double>(std::forward<Args>(args)...);
  }
  template <typename... Args>
  double GetOrAddReal(Args &&...args) {
    return GetOrAdd<double>(std::forward<Args>(args)...);
  }
  template <typename... Args>
  double SetReal(Args &&...args) {
    return Set<double>(std::forward<Args>(args)...);
  }
  template <typename... Args>
  bool GetBoolean(Args &&...args) {
    return Get<bool>(std::forward<Args>(args)...);
  }
  template <typename... Args>
  bool GetOrAddBoolean(Args &&...args) {
    return GetOrAdd<bool>(std::forward<Args>(args)...);
  }
  template <typename... Args>
  bool SetBoolean(Args &&...args) {
    return Set<bool>(std::forward<Args>(args)...);
  }
  template <typename... Args>
  std::string GetString(Args &&...args) {
    return Get<std::string>(std::forward<Args>(args)...);
  }
  template <typename... Args>
  std::string GetOrAddString(Args &&...args) {
    return GetOrAdd<std::string>(std::forward<Args>(args)...);
  }
  template <typename... Args>
  std::string SetString(Args &&...args) {
    return Set<std::string>(std::forward<Args>(args)...);
  }

  template <typename T, typename... Args>
  T Set(const std::string &block, const std::string &name, const T &value,
        const std::optional<std::string> &docstring = std::optional<std::string>{}) {
    return SetPath<T>(ParameterPath(block, name), value, docstring);
  }
  template <typename T>
  T SetPath(const std::string &path, const T &value,
            const std::optional<std::string> &docstring = std::optional<std::string>{}) {
    // Check and error
    // BSP: This is not how Parthenon previously behaved, so it's commented
    // if (!parameters_.at_path(path)) {
    //   std::stringstream msg;
    //   msg << "### FATAL ERROR in function [ParameterInput::Get]" << std::endl
    //       << "Parameter name '" << path << "' not found";
    //   PARTHENON_FAIL(msg);
    // }

    // This is the default, if nothing else is, so record it
    CheckAndUpdateQueries_<T>(path, value, std::vector<T>{}, docstring);

    // We still call AddParameter_, to overwrite the origin
    AddParameter_(parameters_, path, value, OriginType::SetInCode);

    // Convert string to integer and return value
    return value;
  }

  template <typename T>
  T Get(const std::string &block, const std::string &name,
        const std::optional<std::string> &docstring = std::optional<std::string>{}) {
    return GetPath<T>(ParameterPath(block, name), docstring);
  }
  template <typename T, typename... Args>
  T Get(const ParameterRef &r, Args &&...args) {
    return GetPath<T>(r.CanonicalPath(), std::forward<Args>(args)...);
  }

  template <typename T>
  T GetPath(const std::string &path,
            const std::optional<std::string> &docstring = std::optional<std::string>{}) {
    // Check and error
    // TODO(BSP) better compile-time error if type isn't supported by toml++?
    // maybe || !parameters_.is<T>() later? Maybe tweak as<T>()?
    if (!parameters_.at_path(path)) {
      std::stringstream msg;
      msg << "### FATAL ERROR in function [ParameterInput::GetPath]" << std::endl
          << "Parameter name '" << path << "' not found";
      PARTHENON_FAIL(msg);
    }

    CheckAndUpdateQueries_<T>(path, docstring);

    if constexpr (std::is_same<T, toml::table>::value) {
      if (parameters_.at_path(path).is<toml::table>()) {
        return parameters_.at_path(path).ref<toml::table>();
      } else {
        std::stringstream msg;
        msg << "### FATAL ERROR in function [ParameterInput::GetPath]" << std::endl
            << "Parameter name '" << path << "' is of the wrong type" << std::endl
            << "Value: " << parameters_.at_path(path) << " ("
            << parameters_.at_path(path).type() << ")" << std::endl;
        PARTHENON_FAIL(msg);
      }
    } else {
      if (auto val = parameters_.at_path(path).value<T>(); val) {
        return *val;
      } else {
        std::stringstream msg;
        msg << "### FATAL ERROR in function [ParameterInput::GetPath]" << std::endl
            << "Parameter name '" << path << "' is of the wrong type" << std::endl
            << "Value: " << parameters_.at_path(path) << " ("
            << parameters_.at_path(path).type() << ")" << std::endl;
        PARTHENON_FAIL(msg);
      }
    }
  }

  template <typename T>
  T GetOrAdd(const std::string &block, const std::string &name, const T &value,
             const std::optional<std::string> &docstring = std::optional<std::string>{}) {
    return GetOrAddPath<T>(ParameterPath(block, name), value, docstring);
  }
  template <typename T>
  T GetOrAdd(const std::string &block, const std::string &name, const T &value,
             const std::vector<T> allowed_values,
             const std::optional<std::string> &docstring = std::optional<std::string>{}) {
    return GetOrAddPath<T>(ParameterPath(block, name), value, allowed_values, docstring);
  }
  template<typename T, typename... Args>
  T GetOrAdd(const std::string &block, const std::string &name, const ParameterRef &ref,
             Args... &&args) {
    return GetOrAddPath<T>(ParameterPath(block, name), ref, std::forward<Args>(args)...);
  }

  template <typename T>
  T GetOrAddPath(
      const std::string &path, const T &value,
      const std::optional<std::string> &docstring = std::optional<std::string>{}) {
    CheckAndUpdateQueries_<T>(path, value, std::vector<T>{}, docstring);
    if (!parameters_.at_path(path)) {
      AddParameter_(parameters_, path, value, OriginType::Default);
    }
    return GetPath<T>(path);
  }
  template <typename T>
  T GetOrAddPath(
      const std::string &path, const T &value, const std::vector<T> allowed_values,
      const std::optional<std::string> &docstring = std::optional<std::string>{}) {
    // Check allowed values if non-empty
    if (!allowed_values.empty()) CheckAllowedValues_(path, value, allowed_values);

    // Update docs with allowed values
    CheckAndUpdateQueries_<T>(path, value, allowed_values, docstring);

    return GetOrAddPath(path, value);
  }
  template <typename T>
  T GetOrAddPath(
      const std::string &path, const T &value,
      const std::optional<std::string> &docstring = std::optional<std::string>{}) {
    CheckAndUpdateQueries_<T>(path, value, std::vector<T>{}, docstring);
    if (!parameters_.at_path(path)) {
      AddParameter_(parameters_, path, value, OriginType::Default);
    }
    return GetPath<T>(path);
  }
  template <typename T, typename... Args>
  T GetOrAddPath(const std::string &path, const ParameterRef &ref, Args... &&args) {
    auto value = GetPath<T>(ref.CanonicalPath());
    auto ret = GetOrAddPath<T>(path, value, std::forward<Args>(args)...);
    SetQueryDependency_(path, ref);
    return ret;
  }

  template <typename T>
  std::vector<T>
  GetVector(const std::string &block, const std::string &name,
            const std::optional<std::string> &docstring = std::optional<std::string>{}) {
    return GetVectorPath<T>(ParameterPath(block, name), docstring);
  }

  template <typename T>
  std::vector<T> GetVectorPath(
      const std::string &path,
      const std::optional<std::string> &docstring = std::optional<std::string>{}) {
    // Check and error
    // TODO(BSP) type checking of contents or singleton
    if (!parameters_.at_path(path)) {
      std::stringstream msg;
      msg << "### FATAL ERROR in function [ParameterInput::GetVectorPath]" << std::endl
          << "Parameter name '" << path << "' not found";
      PARTHENON_FAIL(msg);
    }

    CheckAndUpdateQueries_<toml::array>(path, docstring);

    std::vector<T> ret;
    // Handle single elements like 1-element arrays wherever they appear
    // We have *no way* of knowing beforehand what's an array when parsing
    if (!parameters_.at_path(path).is_array()) {
      if constexpr (std::is_same<T, int>::value) {
        ret.push_back(static_cast<int>(parameters_.at_path(path).ref<int64_t>()));
      } else {
        ret.push_back(parameters_.at_path(path).ref<T>());
      }
    } else {
      for (const auto &el : *parameters_.at_path(path).as_array()) {
        if constexpr (std::is_same<T, int>::value) {
          ret.push_back(static_cast<int>(el.ref<int64_t>()));
        } else {
          ret.push_back(el.ref<T>());
        }
      }
    }

    return ret;
  }

  template <typename T>
  std::vector<T> GetOrAddVector(
      const std::string &block, const std::string &name, std::vector<T> def,
      const std::optional<std::string> &docstring = std::optional<std::string>{}) {
    return GetOrAddVectorPath<T>(ParameterPath(block, name), def, docstring);
  }
  template <typename T>
  std::vector<T> GetOrAddVector(
      const std::string &block, const std::string &name, const ParameterRef &def,
      const std::optional<std::string> &docstring = std::optional<std::string>{}) {
    auto defval = GetVectorPath<T>(def.CanonicalPath());
    auto ret = GetOrAddVector<T>(block, name, defval, docstring);
    SetQueryDependency_(GetPath(block, name), def);
    return ret;
  }

  template <typename T>
  std::vector<T> GetOrAddVectorPath(
      const std::string &path, std::vector<T> def,
      const std::optional<std::string> &docstring = std::optional<std::string>{}) {
    // Always load defaults into an array, for below
    auto def_array = toml::array();
    for (auto el : def)
      def_array.push_back(el);

    if (!parameters_.at_path(path)) {
      // This mimics "AddParameter_" but specifically for arrays
      InsertOrAssignPath_(parameters_, path, def_array);
      UpdateQueryProvenance_(path, OriginType::Default);
    }

    CheckAndUpdateQueries_<toml::array>(path, def_array, std::vector<toml::array>{},
                                        docstring);

    return GetVectorPath<T>(path);
  }
  template <typename T>
  std::vector<T> GetOrAddVectorPath(
      const std::string &path, const ParameterRef &def,
      const std::optional<std::string> &docstring = std::optional<std::string>{}) {
    auto defval = GetVectorPath<T>(def.CanonicalPath());
    auto ret = GetOrAddVectorPath<T>(path, defval, docstring);
    SetQueryDependency_(path, def);
    return ret;
  }
  // TODO(BSP) SetVector/Path?

 private:
  // Alloc 1MB temporarily, to avoid ever thinking about this again
  static constexpr int max_input_filesize_ = 1024 * 1024;
  toml::table parameters_;
  // We will want to iterate through the record in lexicographic
  // order, so this needs to be an ordered map
  std::map<std::string, QueryRecord> queries_;

  toml::table LegacyParse(std::istream &is, std::string fname = "");
  bool LegacyParseLine(std::string line, std::string &name, std::string &value);
  void Merge(toml::table &a, const toml::table &b, bool check_dups);

  void recursive_get_paths(const toml::table &a, toml::path prefix, const toml::key &key,
                           std::vector<toml::path> &paths) const;

  template <typename T, template <class...> class Container_t, class... extra>
  void CheckAllowedValues_(const std::string &path, const T &val,
                           Container_t<T, extra...> allowed) {
    bool found = std::any_of(allowed.begin(), allowed.end(),
                             [&](const T &t) { return (t == val); });
    if (!found) {
      std::stringstream msg;
      msg << "### FATAL ERROR in function [ParameterInput::Get]\n"
          << "Parameter '" << path << "' must be one of the following values:\n";
      for (const auto &v : allowed) {
        msg << v << " ";
      }
      msg << std::endl;
      PARTHENON_THROW(msg);
    }
  }

  template <typename T>
  inline void recursive_merge(toml::table &a, const toml::table &b, const toml::key &key,
                              T &&el, bool check_dups) {
    if (b[key].is<toml::table>()) {
      a.insert(key, toml::table());
      const toml::table &bchild = b[key].ref<toml::table>();
      toml::table &achild = a[key].ref<toml::table>();
      bchild.for_each([&](const toml::key &key, auto &&el) {
        recursive_merge(achild, bchild, key, el, check_dups);
      });
    } else {
      auto [itr, success] = a.insert(key, el);
      if (!success) {
        if (check_dups) {
          // TODO(BSP) can we print full path here instead of key?
          std::stringstream msg;
          msg << "### ERROR in parameter parsing\n"
              << "Parameter '" << key << "' is duplicate!\n"
              << "Previous definition: " << a[key] << " new definition: " << b[key]
              << std::endl;
          PARTHENON_THROW(msg);
        }
        // Insert over the existing key
        a.insert_or_assign(key, el);
      }
    }
  }

  template <typename T>
  void InsertOrAssignPath_(toml::table &tab, const std::string &path, const T &value) {
    if (path == "") return;
    // Recursively create the tables in the path
    toml::path fullpath = toml::path(path);
    toml::path parent = fullpath.parent();
    if (!tab.at_path(parent)) {
      InsertOrAssigPath_(tab, parent.str(), toml::table());
    }
    // Now we know parent exists (or is the root), so insert just the leaf key
    if (parent.str() == "") {
      tab.insert_or_assign(fullpath.leaf().str(), value);
    } else {
      tab.at_path(parent).ref<toml::table>().insert_or_assign(fullpath.leaf().str(),
                                                              value);
    }
  }

  template <typename T>
  void AddParameter_(toml::table &tbl, const std::string &path, const T &value,
                     OriginType og, bool check_dups = false) {
    // If it's already got a type, just add it
    if constexpr (!std::is_same<T, std::string>::value) {
      InsertOrAssignPath_(tbl, path, value);
    } else {
      // Anything we know is a string: the code says, so, it contains quotes...
      if (og == OriginType::Default || og == OriginType::SetInCode ||
          std::count(value.begin(), value.end(), '\"')) {
        InsertOrAssignPath_(tbl, path, value);
      } else {
        // Otherwise, a "string" might need to be something else internally
        // Parse it with toml++ and see what pops out
        toml::table new_tbl;
        std::string v = value;
        v.erase(std::remove(v.begin(), v.end(), ' '), v.end());
        v.erase(std::remove(v.begin(), v.end(), '\''), v.end());
        if (std::count(v.begin(), v.end(), ',')) {
          // Record an array by adding the necessary TOML
          try {
            v = std::regex_replace(v, std::regex(","), ", ");
            new_tbl = toml::parse(path + " = [" + v + "]");
          } catch (const toml::parse_error &err) {
            v = std::regex_replace(v, std::regex(", "), "\", \"");
            new_tbl = toml::parse(path + " = [\"" + v + "\"]");
          }
        } else {
          // Record parameter
          try {
            new_tbl = toml::parse(path + " = " + v);
          } catch (const toml::parse_error &err) {
            try {
              // If stod would have taken this,
              double v_parsed = std::stod(v.c_str());
              // ...it's because of the 1.eX vs 1.0eX thing, so replace
              v = std::regex_replace(v, std::regex("([0-9])[.]e([+-0-9])"), "$1.0e$2");
              // then parse
              new_tbl = toml::parse(path + " = " + v);
            } catch (const std::invalid_argument &err) {
              new_tbl = toml::parse(path + " = \"" + v + "\"");
            } catch (const toml::parse_error &err) {
              new_tbl = toml::parse(path + " = \"" + v + "\"");
            }
          }
        }

        if (tbl.empty()) {
          tbl = new_tbl;
        } else {
          // Then merge our newly parsed (typed) values into the table
          // Optionally check for duplicates depending on where called
          Merge(tbl, new_tbl, check_dups);
        }
      }
    }

    UpdateQueryProvenance_(path, og);
  }

  // JMM: Using std::optional here aggressively to simplify overload
  // and default parameter logic logic
  template <typename T, template <class...> class Container_t, class... extra>
  void CheckAndUpdateQueries_(const std::string &path, const std::optional<T> &defval,
                              Container_t<T, extra...> allowed_vals,
                              const std::optional<std::string> &docstring) {
    if constexpr (is_sortable_v<decltype(allowed_vals)>) {
      if (allowed_vals.size() > 0) {
        std::sort(std::begin(allowed_vals), std::end(allowed_vals));
      }
    }
    if (queries_.count(path) > 0) {
      QueryRecord &record = queries_.at(path);
      if (defval.has_value()) {
        if (!record.default_value.has_value()) {
          if (record.origin_type == OriginType::SetInCode) {
            // This was set with Set* and we should respect it. Add
            // the new default and move on.
            record.default_value = defval.value();
            record.default_value_str = record.ToString(defval.value());
          } else {
            // JMM: Forbid setting a default value after requesting but
            // allow requesting without a default if a default has
            // already been set.  I know this is unpleasantly stateful,
            // but we do this in a few places in the code.
            std::stringstream msg;
            msg << "Input parameter " << path
                << " called previously without a default value and now called with
                one."
                << " If a default value is used, the first call must always set one."
                << std::endl;
            PARTHENON_THROW(msg);
          }
        } else if (defval.value() != std::any_cast<T>(record.default_value)) {
          std::stringstream msg;
          msg << "Input parameter " << path
              << " has at least two inconsistent default values. "
              << "The ones I detected are " << defval.value() << " and "
              << std::any_cast<T>(record.default_value) << std::endl;
          PARTHENON_THROW(msg);
        }
      }
      // Allowed values are checked after a query, so this function
      // will be called twice: once with no allowed values and once
      // with them. This check ensures that validation for allowed
      // values only happens if they're both active.
      if ((allowed_vals.size() > 0) && (record.allowed_values.size() > 0)) {
        PARTHENON_REQUIRE_THROWS(allowed_vals.size() == record.allowed_values.size(),
                                 "Allowed values must be consistently shaped");
        std::size_t i = 0;
        for (const auto &allowed : allowed_vals) {
          PARTHENON_REQUIRE_THROWS(allowed ==
                                       std::any_cast<T>(record.allowed_values[i++]),
                                   "Allowed values must be consistent");
        }
      } else if (allowed_vals.size() > 0) {
        for (const auto &allowed : allowed_vals) {
          record.allowed_values.push_back(std::any(allowed));
        }
      }
      // if two inconsistent docstrings exist, complain
      if (record.docstring.has_value() && docstring.has_value() &&
          (record.docstring.value() != docstring.value())) {
        std::stringstream msg;
        msg << "Input parameter " << path
            << " has inconsistent docstrings. The strings are:\n"
            << record.docstring.value() << "\nand\n"
            << docstring.value() << std::endl;
        PARTHENON_THROW(msg);
      } else if (docstring.has_value()) {
        // if the new query contains a docstring but the record does
        // not, add the docstring
        record.docstring = docstring; // record is a reference
      }
      // if the record contains a docstring but the new query does
      // not, do nothing
      // if neither contains a docstring, do nothing
    } else {
      QueryRecord record;
      if (defval.has_value()) {
        record.default_value = defval.value();
        record.default_value_str = record.ToString(defval.value());
      } else {
        record.default_value_str = "";
      }
      for (const auto &allowed : allowed_vals) {
        record.allowed_values.push_back(std::any(allowed));
        record.allowed_vals_str.push_back(record.ToString(allowed));
      }
      record.docstring = docstring; // might be empty
      queries_[path] = record;
    }
  }
  template <typename T>
  void CheckAndUpdateQueries_(const std::string &path,
                              const std::optional<std::string> &docstring) {
    CheckAndUpdateQueries_<T>(path, std::optional<T>{}, std::vector<T>{}, docstring);
  }
  void UpdateQueryProvenance_(const std::string path, OriginType og) {
    PARTHENON_REQUIRE_THROWS(queries_.count(path),
                             "Query for path " + path + " not found.");
    queries_.at(path).origin_type = og;
  }
  void SetQueryDependency_(const std::string &path, const ParameterRef &ref) {
    queries_.at(path).default_value_str = ref.CanonicalPath();
  }
};

} // namespace parthenon

// JMM: Believe it or not, this is the recommended way to overload hash functions
// See: https://en.cppreference.com/w/cpp/utility/hash
namespace std {
// We hash the string representation of parameters_, which is what gets written,
// and thus what needs to be consistent between ranks.
template <>
struct hash<parthenon::ParameterInput> {
  std::size_t operator()(const parthenon::ParameterInput &in) {
    std::stringstream ss;
    ss << in.GetAll();
    return std::hash<std::string>()(ss.str());
  }
};
} // namespace std

#endif // PARAMETER_INPUT_HPP_
