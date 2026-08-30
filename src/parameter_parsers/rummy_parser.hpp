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

enum class InputParserPolicy { Auto, NativeOnly, RummyOnly };
enum class RummyMode { Simple, FullLoose, FullStrict };

// Explicit parser selection. Unlike the legacy InputDeckType API this keeps
// format detection separate from the Rummy implementation selected after a
// Rummy deck has been chosen.
struct InputDeckOptions {
  InputParserPolicy parser = InputParserPolicy::Auto;
  RummyMode rummy_mode = RummyMode::Simple;
  std::string schema_path;
};

struct RummyRestartState {
  static constexpr int VERSION = 1;
  int version = VERSION;
  std::string mode;
  std::string source;
};

InputDeckType ToInputDeckType(RummyMode mode);

std::unique_ptr<Rummy::DeckBase>
LoadParameterFromRummy(ParameterInput &input, const std::vector<std::string> &files,
                       const std::vector<std::string> &mods, const bool is_restart,
                       InputDeckType deck_type = InputDeckType::RummySimple,
                       const std::string &schema_path = "");
std::unique_ptr<Rummy::DeckBase>
LoadParameterFromRummy(ParameterInput &input, const std::vector<std::string> &files,
                       const std::vector<std::string> &mods, const bool is_restart,
                       InputDeckType deck_type, std::istream &schema_stream);
std::unique_ptr<Rummy::DeckBase>
LoadParameterFromRummy(ParameterInput &pin, std::istream &ss, const bool sync,
                       InputDeckType deck_type = InputDeckType::RummySimple,
                       const std::string &schema_path = "");
std::unique_ptr<Rummy::DeckBase>
LoadParameterFromRummy(ParameterInput &pin, std::istream &ss, const bool sync,
                       InputDeckType deck_type, std::istream &schema_stream);
std::unique_ptr<Rummy::DeckBase>
LoadParameterFromRummyRestart(ParameterInput &pin, const std::string &restart_source,
                              const std::vector<std::string> &files,
                              const std::vector<std::string> &mods,
                              InputDeckType deck_type);
RummyRestartState MakeRummyRestartState(const ParameterInput &pin,
                                        const Rummy::DeckBase &deck);
InputDeckType RummyRestartModeToDeckType(const std::string &mode);
void AddRummyParameters(ParameterInput &pin, Rummy::DeckBase &deck);
void SyncDeckFromStorage(const ParameterInput &pin, Rummy::DeckBase &deck);
bool IsRummyFormat(const std::string &filename);
bool IsRummyFormat(std::istream &is, const bool command_line);
} // namespace parthenon
#endif // PARAMETER_PARSERS_RUMMY_PARSER_HPP_
