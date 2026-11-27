//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2026 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// This file was made in part with generative AI.

#ifndef OUTPUTS_OPENPMD_NAMES_HPP_
#define OUTPUTS_OPENPMD_NAMES_HPP_

#include <algorithm>
#include <string_view>

namespace parthenon::OpenPMDUtils {

constexpr bool IsValidNameCharacter(char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
         c == '_';
}

inline bool IsValidName(std::string_view name) {
  return !name.empty() && std::all_of(name.begin(), name.end(), IsValidNameCharacter);
}

} // namespace parthenon::OpenPMDUtils

#endif // OUTPUTS_OPENPMD_NAMES_HPP_
