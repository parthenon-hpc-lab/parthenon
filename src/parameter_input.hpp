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
#ifndef PARAMETER_INPUT_HPP_
#define PARAMETER_INPUT_HPP_
//! \file parameter_input.hpp
//  \brief definition of class ParameterInput
// Contains data structures used to store, and functions used to access, parameters
// read from the input file.  See comments at start of parameter_input.cpp for more
// information on the Athena++ input file format.

#include <algorithm>
#include <cstddef>
#include <ostream>
#include <regex>
#include <string>
#include <utility> // for std::forward
#include <vector>

#include <toml.hpp>

#include "config.hpp"
#include "defs.hpp"
#include "outputs/io_wrapper.hpp"
#include "utils/hash.hpp"
#include "utils/string_utils.hpp"

namespace parthenon {

enum class ParameterOrigin { restart, input, cmdline, code, defaultvalue };

//----------------------------------------------------------------------------------------
//! \class ParameterInput
//  \brief data and definitions of functions used to store and access input parameters
//  Functions are implemented in parameter_input.cpp

class ParameterInput {
  friend class std::hash<ParameterInput>;

 public:
  // constructor/destructor
  ParameterInput() {
    parameters_ = toml::table();
    origins_ = toml::table();
  }
  explicit ParameterInput(std::string input_filename) {
    parameters_ = toml::table();
    origins_ = toml::table();
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
  void CheckRequired(const std::string &block, const std::string &name);
  void CheckRequired(const std::string &path);
  void CheckDesired(const std::string &block, const std::string &name);
  void CheckDesired(const std::string &path);
  ParameterOrigin GetOrigin(const std::string &block, const std::string &name);
  ParameterOrigin GetOrigin(const std::string &path);
  std::string GetOriginFile(const std::string &block, const std::string &name);
  std::string GetOriginFile(const std::string &path);
  toml::table Blocks();
  toml::table Blocks(const char *path);
  toml::table Blocks(std::string &path);
  const toml::table GetAll() const;
  const toml::table GetAllOrigins() const;

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
  template <typename... Args>
  Real GetReal(Args &&...args) {
    return Get<Real>(std::forward<Args>(args)...);
  }
  template <typename... Args>
  Real GetOrAddReal(Args &&...args) {
    return GetOrAdd<Real>(std::forward<Args>(args)...);
  }
  template <typename... Args>
  Real SetReal(Args &&...args) {
    return Set<Real>(std::forward<Args>(args)...);
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
        Args &&...args) {
    return SetPath<T>(Path_(block, name), value, std::forward<Args>(args)...);
  }

  template <typename T, typename... Args>
  T SetPath(const std::string &path, const T &value, Args &&...args) {
    // Check and error
    // Can we someday add || !parameters_.is<T>()?
    // This is not how Parthenon previously behaved, so it's commented
    // if (!parameters_.at_path(path)) {
    //   std::stringstream msg;
    //   msg << "### FATAL ERROR in function [ParameterInput::Get]" << std::endl
    //       << "Parameter name '" << path << "' not found";
    //   PARTHENON_FAIL(msg);
    // }

    // We still call AddParameter_, to overwrite the origin
    AddParameter_(parameters_, path, value, ParameterOrigin::code);

    // std::cerr << "Setting " << path << " to " << ss_value.str() << std::endl;

    // Convert string to integer and return value
    return value;
  }

  template <typename T, typename... Args>
  T Get(const std::string &block, const std::string &name, Args &&...args) {
    return GetPath<T>(Path_(block, name), std::forward<Args>(args)...);
  }

  template <typename T, typename... Args>
  T GetPath(const std::string &path, Args &&...args) {
    // Check and error
    // TODO(BSP) better compile-time error if type isn't supported by toml++?
    // maybe || !parameters_.is<T>() later? Maybe tweak as<T>()?
    if (!parameters_.at_path(path)) {
      std::stringstream msg;
      msg << "### FATAL ERROR in function [ParameterInput::GetPath]" << std::endl
          << "Parameter name '" << path << "' not found";
      PARTHENON_FAIL(msg);
    }

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

  template <typename T, typename... Args>
  T GetOrAdd(const std::string &block, const std::string &name, const T &value,
             const std::vector<T> allowed_values = {}, Args &&...args) {
    return GetOrAddPath<T>(Path_(block, name), value, allowed_values,
                           std::forward<Args>(args)...);
  }

  template <typename T, typename... Args>
  T GetOrAddPath(const std::string &path, const T &value,
                 const std::vector<T> allowed_values = {}, Args &&...args) {
    if (!parameters_.at_path(path)) {
      AddParameter_(parameters_, path, value, ParameterOrigin::defaultvalue);
    }
    // TODO(BSP) can we pass an enum and make its contents the allowed_values?
    if (!allowed_values.empty()) CheckAllowedValues_(path, value, allowed_values);

    return GetPath<T>(path, std::forward<Args>(args)...);
  }

  template <typename T, typename... Args>
  std::vector<T> GetVector(const std::string &block, const std::string &name,
                           Args &&...args) {
    return GetVectorPath<T>(Path_(block, name), std::forward<Args>(args)...);
  }

  template <typename T, typename... Args>
  std::vector<T> GetVectorPath(const std::string &path, Args &&...args) {
    // Check and error
    // TODO(BSP) type checking of contents or singleton
    if (!parameters_.at_path(path)) {
      std::stringstream msg;
      msg << "### FATAL ERROR in function [ParameterInput::GetVectorPath]" << std::endl
          << "Parameter name '" << path << "' not found";
      PARTHENON_FAIL(msg);
    }
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

  template <typename T, typename... Args>
  std::vector<T> GetOrAddVector(const std::string &block, const std::string &name,
                                std::vector<T> def, Args &&...args) {
    return GetOrAddVectorPath<T>(Path_(block, name), def, std::forward<Args>(args)...);
  }

  template <typename T, typename... Args>
  std::vector<T> GetOrAddVectorPath(const std::string &path, std::vector<T> def,
                                    Args &&...args) {
    if (!parameters_.at_path(path)) {
      // This mimics "AddParameter_" but specifically for arrays
      InsertOrAssignPath_(parameters_, path, toml::array());
      InsertOrAssignPath_(origins_, path, "default");
      for (auto el : def)
        parameters_.at_path(path).as_array()->push_back(el);
    }
    return GetVectorPath<T>(path);
  }
  // TODO(BSP) SetVector/Path?

 private:
  toml::table LegacyParse(std::istream &is, std::string fname = "");
  bool LegacyParseLine(std::string line, std::string &name, std::string &value);

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
  std::string Path_(const std::string block, const std::string name) {
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

  template <typename T>
  inline void recursive_merge(toml::table &a, const toml::table &b, const toml::key &key,
                              T &&el, bool check_dups) {
    // std::cerr << a << "\ninserted:\n" << el << std::endl;
    if (b[key].is<toml::table>()) {
      a.insert(key, toml::table());
      // std::cerr << "recurse" << std::endl;
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
  inline toml::table &Merge(toml::table &a, const toml::table &b, bool check_dups) {
    b.for_each([&](const toml::key &key, auto &&el) {
      recursive_merge(a, b, key, el, check_dups);
    });
    return a;
  }

  inline void recursive_set_origin(toml::table &a, const toml::table &b,
                                   const toml::key &key, const std::string &origin) {
    // std::cerr << a << "\nsetorigin:\n" << origin << std::endl;
    if (b[key].is<toml::table>()) {
      a.insert(key, toml::table());
      // std::cerr << "recurse" << std::endl;
      const toml::table &bchild = b[key].ref<toml::table>();
      toml::table &achild = a[key].ref<toml::table>();
      bchild.for_each([&](const toml::key &key, auto &&el) {
        recursive_set_origin(achild, bchild, key, origin);
      });
    } else {
      a.insert_or_assign(key, origin);
    }
  }
  inline toml::table &SetOrigin(toml::table &a, toml::table &b,
                                const std::string origin) {
    b.for_each([&](const toml::key &key, auto &&el) {
      recursive_set_origin(a, b, key, origin);
    });
    return a;
  }

  template <typename T>
  void InsertOrAssignPath_(toml::table &tab, const std::string &path, const T &value) {
    if (path == "") return;
    // Recursively create the tables in the path
    toml::path fullpath = toml::path(path);
    toml::path parent = fullpath.parent();
    if (!tab.at_path(parent)) {
      InsertOrAssignPath_(tab, parent.str(), toml::table());
    }
    // Now we know parent exists (or is the root), so insert just the leaf key
    if (parent.str() == "") {
      tab.insert_or_assign(fullpath.leaf().str(), value);
    } else {
      tab.at_path(parent).ref<toml::table>().insert_or_assign(fullpath.leaf().str(),
                                                              value);
    }
  }

  // TODO(BSP) Add overload natively accepting vectors
  template <typename T>
  void AddParameter_(toml::table &tbl, const std::string &path, const T &value,
                     ParameterOrigin og, bool check_dups = false,
                     std::string originfile = "") {
    // If it's already got a type, just add it
    if constexpr (!std::is_same<T, std::string>::value) {
      InsertOrAssignPath_(tbl, path, value);
    } else {
      // Anything we know is a string: the code says, so, it contains quotes...
      if (og == ParameterOrigin::defaultvalue || og == ParameterOrigin::code ||
          std::count(value.begin(), value.end(), '\"')) {
        InsertOrAssignPath_(tbl, path, value);
      } else {
        // Otherwise, a "string" might need to be something else internally
        // Parse it with toml++ and see what pops out
        toml::table new_tbl;
        std::string v = value;
        v.erase(std::remove(v.begin(), v.end(), ' '), v.end());
        v.erase(std::remove(v.begin(), v.end(), '\''), v.end());
        // std::cerr << "INSERTING: " << path << "=" << v << std::endl;
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
            new_tbl = toml::parse(path + " = \"" + v + "\"");
          }
        }
        // std::cerr << "Adding new table:" << std::endl << new_tbl << std::endl;
        // std::cerr << "To existing table:" << std::endl << tbl << std::endl;

        if (tbl.empty()) {
          tbl = new_tbl;
        } else {
          // Then merge our newly parsed (typed) values into the table
          // Optionally check for duplicates depending on where called
          Merge(tbl, new_tbl, check_dups);
        }
      }
    }

    // Record provenance
    switch (og) { // restart, input, cmdline, code, defaultvalue
    case ParameterOrigin::restart:
      InsertOrAssignPath_(origins_, path, "restart");
      break;
    case ParameterOrigin::cmdline:
      InsertOrAssignPath_(origins_, path, "cmdline");
      break;
    case ParameterOrigin::code:
      InsertOrAssignPath_(origins_, path, "code");
      break;
    case ParameterOrigin::defaultvalue:
      InsertOrAssignPath_(origins_, path, "default");
      break;
    case ParameterOrigin::input:
      InsertOrAssignPath_(origins_, path, originfile);
      break;
    }
  }

  // Alloc 1MB temporarily, to avoid ever thinking about this again
  static constexpr int max_input_filesize_ = 1024 * 1024;
  toml::table parameters_, origins_;
  std::vector<toml::table> parameter_lists_;
};
} // namespace parthenon

// JMM: Believe it or not, this is the recommended way to overload hash functions
// See: https://en.cppreference.com/w/cpp/utility/hash
namespace std {
template <>
struct hash<parthenon::ParameterInput> {
  std::size_t operator()(const parthenon::ParameterInput &in) {
    using parthenon::impl::hash_combine;
    std::size_t out = 0;
    out = hash_combine(out, in.GetAll(), in.GetAllOrigins());
    return out;
  }
};

// Be a little evil.  It's good for you.
template <>
struct hash<toml::v3::table> {
  std::size_t operator()(const toml::v3::table &in) {
    std::stringstream ss;
    ss << in;
    return std::hash<std::string>()(ss.str());
  }
};
} // namespace std

#endif // PARAMETER_INPUT_HPP_
