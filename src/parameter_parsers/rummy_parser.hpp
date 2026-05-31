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

#include <istream>
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
enum class InputDeckType {
  Native = 0,
  RummySimple = 1,
  RummyFullLoose = 2,
  RummyFullStrict = 3,
  RummyFullSchema = 4,
};

void LoadParameterFromRummy(ParameterInput &input, const std::vector<std::string> &files,
                            const std::vector<std::string> &mods, const bool is_restart,
                            InputDeckType deck_type = InputDeckType::RummySimple,
                            const std::string &schema_path = "");
void LoadParameterFromRummy(ParameterInput &input, const std::vector<std::string> &files,
                            const std::vector<std::string> &mods, const bool is_restart,
                            InputDeckType deck_type, std::istream &schema_stream);
void LoadParameterFromRummy(ParameterInput &pin, std::istream &ss, const bool sync,
                            InputDeckType deck_type = InputDeckType::RummySimple,
                            const std::string &schema_path = "");
void LoadParameterFromRummy(ParameterInput &pin, std::istream &ss, const bool sync,
                            InputDeckType deck_type, std::istream &schema_stream);
void AddRummyParameters(ParameterInput &pin, Rummy::DeckBase &deck);
void SyncDeckFromStorage(ParameterInput &pin, Rummy::DeckBase &deck);
bool IsRummyFormat(const std::string &filename);
bool IsRummyFormat(std::istream &is, const bool command_line);
} // namespace parthenon
#endif // PARAMETER_PARSERS_RUMMY_PARSER_HPP_
