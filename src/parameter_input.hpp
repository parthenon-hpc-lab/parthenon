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
// read from the input file.  See comments at start of parameter_input.cpp for more
// information on the Athena++ input file format.

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

struct QueryRecord {
  // TODO(JMM): Update this with more provenance information
  enum class OriginType { None, Input, Default, SetInCode };
  OriginType origin_type = OriginType::Input;
  std::string param_type;
  std::any default_value; // std::any::has_value to check if default
                          // val exists
  std::string
      default_value_str; // used for output, so we don't have to mess with types later
  std::vector<std::any> allowed_values;      // size to check if allowed values exist
  std::vector<std::string> allowed_vals_str; // used for output
  std::optional<std::string> docstring; // std::optional::has_value to check if exists
  // JMM: Surely there's a way of doing this automatically?
  // Unfortunately the value in typeid is implementation defined, so
  // we can't pick if it looks nice...
  template <typename T>
  static std::string GetTypeName() {
    if constexpr (std::is_same_v<T, int>) {
      return "int";
    } else if constexpr (std::is_same_v<T, Real>) {
      return "Real";
    } else if constexpr (std::is_same_v<T, std::uint64_t>) {
      return "uint64_t";
    } else if constexpr (std::is_same_v<T, bool>) {
      return "bool";
    } else if constexpr (std::is_same_v<T, std::string>) {
      return "string";
    } else if constexpr (std::is_arithmetic_v<T>) {
      if (Globals::my_rank == 0) {
        PARTHENON_WARN("Unknown arithmetic type! Attempting to use typeid, which is "
                       "implementation defined.");
      }
      T t;
      return typeid(t).name();
    } else if constexpr (std::is_same_v<T, std::vector<typename T::value_type>>) {
      return "std::vector<" + GetTypeName<typename T::value_type>() + ">";
    } else {
      if (Globals::my_rank == 0) {
        PARTHENON_WARN("Unknown non-arithmetic type! Attempting to use typeid, which is "
                       "implementation defined.");
      }
      T t;
      return typeid(t).name();
    }
  }
  template <typename T>
  static std::string GetTypeName(const T &t) {
    return GetTypeName<T>();
  }
  template <typename T>
  void SetTypeName() {
    param_type = GetTypeName<T>();
  }
  template <typename T>
  static std::string ToString(const T &val) {
    std::stringstream ss;
    if constexpr (std::is_same_v<T, Real>) {
      ss.precision(8); // max digits is totally unreadable
      // ss.precision(std::numeric_limits<T>::max_digits10);
    }
    ss << val;
    return ss.str();
  }
};

// This can be used to tell the params infrastructure that the default
// value of one parameter depends on another one
class ParameterInput;
class ParameterRef {
 public:
  friend class ParameterInput;
  ParameterRef(const std::string &block, const std::string &name)
      : block_(block), name_(name) {}
  // TODO(JMM): Change this to TOML when appropriate
  std::string CanonicalPath() const { return block_ + "/" + name_; }

 private:
  // TODO(JMM): Change this to "canonical path" when available
  const std::string block_;
  const std::string name_;
};

//----------------------------------------------------------------------------------------
//! \struct InputLine
//  \brief  node in a singly linked list of parameters contained within 1x input block

struct InputLine {
  std::string param_name;
  std::string param_value; // value of the parameter is stored as a string!
  std::string param_comment;
  InputLine *pnext; // pointer to the next node in this nested singly linked list
};

//----------------------------------------------------------------------------------------
//! \class InputBlock
//  \brief node in a singly linked list of all input blocks contained within input file

class InputBlock {
 public:
  InputBlock() = default;
  ~InputBlock();

  // data
  std::string block_name;
  std::size_t max_len_parname;  // length of longest param_name, for nice-looking output
  std::size_t max_len_parvalue; // length of longest param_value, to format outputs
  InputBlock *pnext; // pointer to the next node in InputBlock singly linked list

  InputLine *pline; // pointer to head node in nested singly linked list (in this block)
  // (not storing a reference to the tail node)

  // functions
  InputLine *GetPtrToLine(std::string name);
};

//----------------------------------------------------------------------------------------
//! \class ParameterInput
//  \brief data and definitions of functions used to store and access input parameters
//  Functions are implemented in parameter_input.cpp

class ParameterInput {
  friend class std::hash<ParameterInput>;

 public:
  // constructor/destructor
  ParameterInput();
  explicit ParameterInput(std::string input_filename);
  ~ParameterInput();

  // data
  InputBlock *pfirst_block; // pointer to head node in singly linked list of InputBlock
  // (not storing a reference to the tail node)

  // functions
  void LoadFromStream(std::istream &is);
  void LoadFromFile(IOWrapper &input);
  void ModifyFromCmdline(int argc, char *argv[]);
  void ParameterDump(std::ostream &os);
  // TODO(JMM): Make this more general?
  void OutputParameterTable(std::ostream &os,
                            const std::regex &block_regex = std::regex("(.*)")) const;

  int DoesParameterExist(const std::string &block, const std::string &name);
  int DoesBlockExist(const std::string &block);
  std::string GetComment(const std::string &block, const std::string &name);
  int GetInteger(
      const std::string &block, const std::string &name,
      const std::optional<std::string> &docstring = std::optional<std::string>{});
  int GetOrAddInteger(
      const std::string &block, const std::string &name, int value,
      const std::optional<std::string> &docstring = std::optional<std::string>{});
  int GetOrAddInteger(
      const std::string &block, const std::string &name, const ParameterRef &value,
      const std::optional<std::string> &docstring = std::optional<std::string>{});
  int SetInteger(
      const std::string &block, const std::string &name, int value,
      const std::optional<std::string> &docstring = std::optional<std::string>{});
  Real
  GetReal(const std::string &block, const std::string &name,
          const std::optional<std::string> &docstring = std::optional<std::string>{});
  Real GetOrAddReal(
      const std::string &block, const std::string &name, Real value,
      const std::optional<std::string> &docstring = std::optional<std::string>{});
  Real GetOrAddReal(
      const std::string &block, const std::string &name, const ParameterRef &value,
      const std::optional<std::string> &docstring = std::optional<std::string>{});
  Real
  SetReal(const std::string &block, const std::string &name, Real value,
          const std::optional<std::string> &docstring = std::optional<std::string>{});
  bool
  GetBoolean(const std::string &block, const std::string &name,
             const std::optional<std::string> &docstring = std::optional<std::string>{});
  bool GetOrAddBoolean(
      const std::string &block, const std::string &name, bool value,
      const std::optional<std::string> &docstring = std::optional<std::string>{});
  bool GetOrAddBoolean(
      const std::string &block, const std::string &name, const ParameterRef &value,
      const std::optional<std::string> &docstring = std::optional<std::string>{});
  bool
  SetBoolean(const std::string &block, const std::string &name, bool value,
             const std::optional<std::string> &docstring = std::optional<std::string>{});

  std::string
  GetString(const std::string &block, const std::string &name,
            const std::optional<std::string> &docstring = std::optional<std::string>{});
  std::string GetOrAddString(
      const std::string &block, const std::string &name, const std::string &value,
      const std::optional<std::string> &docstring = std::optional<std::string>{});
  std::string
  SetString(const std::string &block, const std::string &name, const std::string &value,
            const std::optional<std::string> &docstring = std::optional<std::string>{});
  std::string
  GetString(const std::string &block, const std::string &name,
            const std::vector<std::string> &allowed_values,
            const std::optional<std::string> &docstring = std::optional<std::string>{});
  std::string GetOrAddString(
      const std::string &block, const std::string &name, const std::string &value,
      const std::vector<std::string> &allowed_values,
      const std::optional<std::string> &docstring = std::optional<std::string>{});
  template <typename... Args>
  std::string GetOrAddString(const std::string &block, const std::string &name,
                             const ParameterRef &ref, Args... args) {
    auto defval = Get<std::string>(ref);
    auto ret = GetOrAddString(block, name, defval, std::forward<Args>(args)...);
    SetQueryDependency_(block, name, ref);
    return ret;
  }
  void RemoveParameter(const std::string &block, const std::string &name);
  void CheckRequired(const std::string &block, const std::string &name);
  void CheckDesired(const std::string &block, const std::string &name);
  void CheckOrphans() const;

  template <typename T, typename... Args>
  T GetOrAdd(const std::string &block, const std::string &name, const T &value,
             Args &&...args) {
    if constexpr (std::is_same_v<T, bool>) {
      return GetOrAddBoolean(block, name, value, std::forward<Args>(args)...);
    } else if constexpr (std::is_same_v<T, Real>) {
      return GetOrAddReal(block, name, value, std::forward<Args>(args)...);
    } else if constexpr (std::is_same_v<T, std::string>) {
      return GetOrAddString(block, name, value, std::forward<Args>(args)...);
    } else if (std::is_integral_v<T>) {
      return GetOrAddInteger(block, name, value, std::forward<Args>(args)...);
    } else {
      PARTHENON_THROW("Unknown type\n");
    }
  }
  template <typename T, typename... Args>
  T Get(const std::string &block, const std::string &name, Args &&...args) {
    if constexpr (std::is_same_v<T, bool>) {
      return GetBoolean(block, name, std::forward<Args>(args)...);
    } else if constexpr (std::is_same_v<T, Real>) {
      return GetReal(block, name, std::forward<Args>(args)...);
    } else if constexpr (std::is_same_v<T, std::string>) {
      return GetString(block, name, std::forward<Args>(args)...);
    } else if (std::is_integral_v<T>) {
      return GetInteger(block, name, std::forward<Args>(args)...);
    } else {
      PARTHENON_THROW("Unknown type\n");
    }
  }
  template <typename T, typename... Args>
  T Get(const ParameterRef &r, Args &&...args) {
    return Get<T>(r.block_, r.name_, std::forward<Args>(args)...);
  }
  template <typename T, typename... Args>
  T GetOrAdd(const std::string &block, const std::string &name, const ParameterRef &value,
             Args &&...args) {
    T ret = GetOrAdd<T>(block, name, Get<T>(value.block_, value.name_),
                        std::forward<Args>(args)...);
    SetQueryDependency_(block, name, value);
    return ret;
  }

  template <typename T>
  std::vector<T>
  GetVector(const std::string &block, const std::string &name,
            const std::optional<std::string> &docstring = std::optional<std::string>{}) {
    std::vector<std::string> fields = GetVector_(block, name);
    if constexpr (std::is_same<T, std::string>::value) return fields;

    std::vector<T> ret;
    for (auto &f : fields) {
      if constexpr (std::is_same<T, int>::value) {
        ret.push_back(stoi(f));
      } else if constexpr (std::is_same<T, Real>::value) {
        ret.push_back(atof(f.c_str()));
      } else if constexpr (std::is_same<T, bool>::value) {
        ret.push_back(stob(f));
      }
    }
    CheckAndUpdateQueries_<std::vector<T>>(block, name, docstring);
    return ret;
  }
  template <typename T>
  std::vector<T> GetOrAddVector(
      const std::string &block, const std::string &name, std::vector<T> def,
      const std::optional<std::string> &docstring = std::optional<std::string>{}) {
    CheckAndUpdateQueries_<std::vector<T>>(block, name, def,
                                           std::vector<std::vector<T>>{}, docstring);
    if (DoesParameterExist(block, name)) return GetVector<T>(block, name);

    std::string cname = ConcatVector_(def);
    auto *pb = FindOrAddBlock(block);
    AddParameter(pb, name, cname, "# Default value added at run time");
    return def;
  }
  template <typename T>
  std::vector<T> GetOrAddVector(
      const std::string &block, const std::string &name, const ParameterRef &def,
      const std::optional<std::string> &docstring = std::optional<std::string>{}) {
    auto defval = GetVector<T>(block, name);
    auto ret = GetOrAddVector<T>(block, name, defval, docstring);
    SetQueryDependency_(block, name, def);
    return ret;
  }

 private:
  std::string last_filename_; // last input file opened, to prevent duplicate reads
  // We will want to iterate through the record in lexicographic
  // order, so this needs to be an ordered map
  std::map<std::pair<std::string, std::string>, QueryRecord> queries_;

  InputBlock *FindOrAddBlock(const std::string &name);
  InputBlock *GetPtrToBlock(const std::string &name);
  bool ParseLine(InputBlock *pib, std::string line, std::string &name, std::string &value,
                 std::string &comment);
  void AddParameter(InputBlock *pib, const std::string &name, const std::string &value,
                    const std::string &comment);
  bool stob(std::string val) {
    // check is string contains integers 0 or 1 (instead of true or false) and return
    if (val.compare(0, 1, "0") == 0 || val.compare(0, 1, "1") == 0) {
      return static_cast<bool>(stoi(val));
    }

    // convert string to all lower case
    std::transform(val.begin(), val.end(), val.begin(), ::tolower);
    // Convert string to bool and return value
    bool b;
    std::istringstream is(val);
    is >> std::boolalpha >> b;
    return b;
  }
  template <typename T, template <class...> class Container_t, class... extra>
  void CheckAllowedValues_(const std::string &block, const std::string &name,
                           const T &val, Container_t<T, extra...> allowed) {
    bool found = std::any_of(allowed.begin(), allowed.end(),
                             [&](const T &t) { return (t == val); });
    if (!found) {
      std::stringstream msg;
      msg << "### FATAL ERROR in function [ParameterInput::Get*]\n"
          << "Parameter '" << name << "/" << block
          << "' must be one of the following values:\n";
      for (const auto &v : allowed) {
        msg << v << " ";
      }
      msg << std::endl;
      PARTHENON_THROW(msg);
    }
  }
  std::vector<std::string> GetVector_(const std::string &block, const std::string &name) {
    std::string s = GetString(block, name);
    std::string delimiter = ",";
    size_t pos = 0;
    std::string token;
    std::vector<std::string> variables;
    while ((pos = s.find(delimiter)) != std::string::npos) {
      token = s.substr(0, pos);
      variables.push_back(string_utils::trim(token));
      s.erase(0, pos + delimiter.length());
    }
    variables.push_back(string_utils::trim(s));
    return variables;
  }
  template <typename T>
  std::string ConcatVector_(std::vector<T> &vec) {
    std::stringstream ss;
    const int n = vec.size();
    if (n == 0) return "";

    ss << vec[0];
    for (int i = 1; i < n; i++) {
      ss << "," << vec[i];
    }
    return ss.str();
  }

  // JMM: Using std::optional here aggressively to simplify overload
  // and default parameter logic logic
  template <typename T, template <class...> class Container_t, class... extra>
  void CheckAndUpdateQueries_(const std::string &block, const std::string &name,
                              const std::optional<T> &defval,
                              Container_t<T, extra...> allowed_vals,
                              const std::optional<std::string> &docstring) {
    if constexpr (is_sortable_v<decltype(allowed_vals)>) {
      if (allowed_vals.size() > 0) {
        std::sort(std::begin(allowed_vals), std::end(allowed_vals));
      }
    }
    auto key = std::make_pair(block, name);
    if (queries_.count(key) > 0) {
      QueryRecord &record = queries_.at(key);
      if (defval.has_value()) {
        if (!record.default_value.has_value()) {
          if (record.origin_type == QueryRecord::OriginType::SetInCode) {
            // This was set with Set* and we should respect it. Add
            // the new default and move on.
            record.default_value = defval.value();
            record.default_value_str = record.ToString(defval.value());
          } else if (record.origin_type != QueryRecord::OriginType::Input) {
            // JMM: Forbid setting a default value after requesting but
            // allow requesting without a default if a default has
            // already been set.  I know this is unpleasantly stateful,
            // but we do this in a few places in the code.
            // PG: but only trigger if input does not contain the info
            std::stringstream msg;
            msg << "Input parameter " << block << "/" << name
                << " called previously without a default value and now called with one."
                << " If a default value is used, the first call must always set one."
                << std::endl;
            PARTHENON_THROW(msg);
          }
        } else if (defval.value() != std::any_cast<T>(record.default_value)) {
          std::stringstream msg;
          msg << "Input parameter " << block << "/" << name
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
          record.allowed_vals_str.push_back(record.ToString(allowed));
        }
      }
      // if two inconsistent docstrings exist, complain
      if (record.docstring.has_value() && docstring.has_value() &&
          (record.docstring.value() != docstring.value())) {
        std::stringstream msg;
        msg << "Input parameter " << block << "/" << name
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
      record.SetTypeName<T>();
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
      queries_[key] = record;
    }
  }
  template <typename T>
  void CheckAndUpdateQueries_(const std::string &block, const std::string &name,
                              const std::optional<std::string> &docstring) {
    CheckAndUpdateQueries_<T>(block, name, std::optional<T>{}, std::vector<T>{},
                              docstring);
  }
  void UpdateQueryProvenance_(const std::string &block, const std::string &name,
                              QueryRecord::OriginType origin) {
    auto key = std::make_pair(block, name);
    queries_.at(key).origin_type = origin;
  }
  void SetQueryDependency_(const std::string &block, const std::string &name,
                           const ParameterRef &ref) {
    auto &query = queries_.at(std::make_pair(ref.block_, ref.name_));
    query.default_value_str = ref.CanonicalPath();
  }
};

} // namespace parthenon

// JMM: Believe it or not, this is the recommended way to overload hash functions
// See: https://en.cppreference.com/w/cpp/utility/hash
namespace std {
template <>
struct hash<parthenon::InputLine> {
  std::size_t operator()(const parthenon::InputLine &il) {
    return parthenon::impl::hash_combine(0, il.param_name, il.param_value,
                                         il.param_comment);
  }
};

template <>
struct hash<parthenon::InputBlock> {
  std::size_t operator()(const parthenon::InputBlock &ib) {
    using parthenon::impl::hash_combine;
    std::size_t out =
        hash_combine(0, ib.block_name, ib.max_len_parname, ib.max_len_parvalue);
    for (parthenon::InputLine *pline = ib.pline; pline != nullptr; pline = pline->pnext) {
      out = hash_combine(out, *pline);
    }
    return out;
  }
};

template <>
struct hash<parthenon::ParameterInput> {
  std::size_t operator()(const parthenon::ParameterInput &in) {
    using parthenon::InputBlock;
    using parthenon::impl::hash_combine;
    std::size_t out = 0;
    out = hash_combine(out, in.last_filename_);
    for (InputBlock *pblock = in.pfirst_block; pblock != nullptr;
         pblock = pblock->pnext) {
      out = hash_combine(out, *pblock);
    }
    return out;
  }
};
} // namespace std

#endif // PARAMETER_INPUT_HPP_
