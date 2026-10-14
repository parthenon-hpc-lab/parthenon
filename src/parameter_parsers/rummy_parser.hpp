//========================================================================================
// (C) (or copyright) 2026. Triad National Security, LLC. All rights reserved.
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
// This file was made in part with generative AI.

#ifndef PARAMETER_PARSERS_RUMMY_PARSER_HPP_
#define PARAMETER_PARSERS_RUMMY_PARSER_HPP_

#include <memory>
#include <string>
#include <vector>

#include "parameter_input.hpp"

// Foward declare Rummy::Deck to avoid including the full header in this file
namespace Rummy {
class Deck;
}
namespace parthenon {
void LoadParameterFromRummy(ParameterInput &input, const std::vector<std::string> &files,
                            const std::vector<std::string> &mods, const bool is_restart);
void LoadParameterFromRummy(ParameterInput &pin, std::istream &ss, const bool sync);
void AddRummyParameters(ParameterInput &pin, Rummy::Deck &deck);
void SyncDeckFromStorage(ParameterInput &pin, Rummy::Deck &deck);
bool IsRummyFormat(const std::string &filename);
bool IsRummyFormat(std::istream &is, const bool command_line);
} // namespace parthenon
#endif // PARAMETER_PARSERS_RUMMY_PARSER_HPP_
