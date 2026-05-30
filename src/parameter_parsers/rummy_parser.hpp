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

// Forward declare Rummy deck classes to avoid including full headers here.
namespace Rummy {
class DeckBase;
class SimpleDeck;
class FullDeck;
} // namespace Rummy
namespace parthenon {

// Selects the rummy deck flavor used to parse input files at runtime. Simple
// is the legacy flat key=value parser; Full enables the pips-backed parser
// (control flow, user classes, <Class(name)> instantiation headers, etc.).
enum class RummyDeckType { Simple, Full };

void LoadParameterFromRummy(ParameterInput &input, const std::vector<std::string> &files,
                            const std::vector<std::string> &mods, const bool is_restart,
                            RummyDeckType deck_type = RummyDeckType::Simple);
void LoadParameterFromRummy(ParameterInput &pin, std::istream &ss, const bool sync,
                            RummyDeckType deck_type = RummyDeckType::Simple);
void AddRummyParameters(ParameterInput &pin, Rummy::DeckBase &deck);
void SyncDeckFromStorage(ParameterInput &pin, Rummy::DeckBase &deck);
bool IsRummyFormat(const std::string &filename);
bool IsRummyFormat(std::istream &is, const bool command_line);
} // namespace parthenon
#endif // PARAMETER_PARSERS_RUMMY_PARSER_HPP_
