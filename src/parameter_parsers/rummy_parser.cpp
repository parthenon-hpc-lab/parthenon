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

#include <algorithm>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <rummy/full_deck.hpp>
#include <rummy/simple_deck.hpp>

#include "parameter_input.hpp"
#include "rummy_parser.hpp"

namespace parthenon {

//! \fn ParameterInput::ParamValue RummyCardToParamValue(const Rummy::Card &card)
//   \brief Convert a Rummy Card to a ParameterInput::ParamValue for storage in
//   ParameterInput.
ParamValue RummyCardToParamValue(const Rummy::Card &card) {
  if (card.isBool()) {
    return card.Get<bool>();
  } else if (card.isString()) {
    return card.Get<std::string>();
  } else {
    // Otherwise store as UnresolvedString to preserve full precision
    return UnresolvedString(card.GetString(std::numeric_limits<double>::max_digits10));
  }
}

//! \fn Rummy::Card ParamValueToRummyCard(suit, name, v)
//   \brief Convert a scalar ParamValue to a Rummy::Card.
Rummy::Card ParamValueToRummyCard(const std::string &suit, const std::string &name,
                                  const ParamValue &v) {
  if (std::holds_alternative<bool>(v))
    return Rummy::Card(suit, name, std::get<bool>(v), "");
  if (std::holds_alternative<int>(v))
    return Rummy::Card(suit, name, static_cast<double>(std::get<int>(v)), "");
  if (std::holds_alternative<Real>(v))
    return Rummy::Card(suit, name, static_cast<double>(std::get<Real>(v)), "");
  if (std::holds_alternative<std::string>(v))
    return Rummy::Card(suit, name, std::get<std::string>(v), "");
  // UnresolvedString
  const std::string &raw = std::get<UnresolvedString>(v).value;
  std::string trimmed = SanitizeString(raw);

  std::string lower = trimmed;
  std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

  if (lower == "true") return Rummy::Card(suit, name, true, "");
  if (lower == "false") return Rummy::Card(suit, name, false, "");
  try {
    std::size_t pos;
    double d = std::stod(trimmed, &pos);
    return Rummy::Card(suit, name, d, "");
  } catch (...) {
  }
  return Rummy::Card(suit, name, trimmed, "");
}

namespace {

std::unique_ptr<Rummy::DeckBase> MakeDeck(InputDeckType /*deck_type*/,
                                          std::istream &schema_stream) {
  std::string schema_text((std::istreambuf_iterator<char>(schema_stream)),
                          std::istreambuf_iterator<char>());
  auto schema = Rummy::Schema::FromString(schema_text);
  return std::make_unique<Rummy::FullDeck>(Rummy::FullDeck::Mode::Strict,
                                           std::move(schema));
}

// Construct a deck of the requested type
std::unique_ptr<Rummy::DeckBase> MakeDeck(InputDeckType deck_type,
                                          const std::string &schema_path = "") {
  switch (deck_type) {
  case InputDeckType::RummyFullLoose:
    return std::make_unique<Rummy::FullDeck>(Rummy::FullDeck::Mode::Loose);
  case InputDeckType::RummyFullStrict: {
    if (schema_path.empty()) {
      PARTHENON_FAIL("InputDeckType::RummyFullStrict requires a non-empty schema_path");
    }
    std::ifstream f(schema_path);
    if (!f.is_open()) {
      std::stringstream msg;
      msg << "Could not open schema file '" << schema_path << "'";
      PARTHENON_FAIL(msg);
    }
    return MakeDeck(deck_type, static_cast<std::istream &>(f));
  }
  case InputDeckType::RummySimple:
  default:
    return std::make_unique<Rummy::SimpleDeck>();
  }
}

} // namespace

void LoadParameterFromRummy(ParameterInput &pin, std::istream &ss, const bool sync,
                            InputDeckType deck_type, const std::string &schema_path) {
  auto deck = MakeDeck(deck_type, schema_path);
  if (sync) {
    SyncDeckFromStorage(pin, *deck);
  }
  deck->Build(ss);
  AddRummyParameters(pin, *deck);
}

void LoadParameterFromRummy(ParameterInput &pin, const std::vector<std::string> &files,
                            const std::vector<std::string> &mods, const bool is_restart,
                            InputDeckType deck_type, const std::string &schema_path) {
  auto deck = MakeDeck(deck_type, schema_path);

  const bool no_inputs = files.empty() && mods.empty();
  if (no_inputs) {
    return;
  }

  if (is_restart) {
    // If this is a restart, we need to sync the deck with the existing parameters
    SyncDeckFromStorage(pin, *deck);
  }

  // concatenate all input files and mods into a single stream for parsing
  std::stringstream contents;
  for (const auto &file : files) {
    std::ifstream input_file(file);
    if (input_file.is_open()) {
      contents << input_file.rdbuf() << "\n";
    } else {
      std::stringstream msg;
      msg << "Could not open file '" << file << "'";
      PARTHENON_FAIL(msg);
    }
  }
  for (const auto &mod : mods) {
    contents << mod << " # From command line\n";
  }

  deck->Build(contents);

  AddRummyParameters(pin, *deck);
}

void LoadParameterFromRummy(ParameterInput &pin, std::istream &ss, const bool sync,
                            InputDeckType deck_type, std::istream &schema_stream) {
  auto deck = MakeDeck(deck_type, schema_stream);
  if (sync) {
    SyncDeckFromStorage(pin, *deck);
  }
  deck->Build(ss);
  AddRummyParameters(pin, *deck);
}

void LoadParameterFromRummy(ParameterInput &pin, const std::vector<std::string> &files,
                            const std::vector<std::string> &mods, const bool is_restart,
                            InputDeckType deck_type, std::istream &schema_stream) {
  auto deck = MakeDeck(deck_type, schema_stream);

  const bool no_inputs = files.empty() && mods.empty();
  if (no_inputs) {
    return;
  }

  if (is_restart) {
    SyncDeckFromStorage(pin, *deck);
  }

  std::stringstream contents;
  for (const auto &file : files) {
    std::ifstream input_file(file);
    if (input_file.is_open()) {
      contents << input_file.rdbuf() << "\n";
    } else {
      std::stringstream msg;
      msg << "Could not open file '" << file << "'";
      PARTHENON_FAIL(msg);
    }
  }
  for (const auto &mod : mods) {
    contents << mod << " # From command line\n";
  }

  deck->Build(contents);
  AddRummyParameters(pin, *deck);
}

void AddRummyParameters(ParameterInput &pin, Rummy::DeckBase &deck) {
  static const std::regex kVectorCardPattern(R"(^(.+)\[(\d+)\]$)");

  // If the deck is a FullDeck, capture per-suit class metadata so callers can
  // query blocks by their pips class via ParameterInput::GetBlocksOfClass.
  auto *full_deck = dynamic_cast<Rummy::FullDeck *>(&deck);

  for (const auto &suit_name : deck.GetSuitsInOrder()) {
    const std::string &block_name = suit_name;
    const auto &suit_cards = deck.GetCardsInOrder(suit_name);
    for (const auto &card_name : suit_cards) {
      // match for vector
      if (deck.IsCardVector(suit_name, card_name)) {
        std::vector<std::string> comments;
        auto elements = deck.GetVector<std::string>(suit_name, card_name, comments);
        std::string joined;
        std::string joined_comments;
        for (std::size_t i = 0; i < elements.size(); ++i) {
          if (comments[i] != "") {
            if (i > 0) {
              joined_comments += " ";
            }
            joined_comments += comments[i];
          }
          if (i > 0) {
            joined += ",";
          }
          joined += elements[i];
        }
        // Rummy stores comments without '#'
        std::string comment;
        if (!joined_comments.empty()) comment = "# " + joined_comments;
        pin.AddParsedParameter(block_name, card_name, UnresolvedString(joined), comment);
      } else {
        auto &card = deck.GetCard(suit_name, card_name);
        std::string comment;
        if (!card.GetComment().empty()) comment = "# " + card.GetComment();
        pin.AddParsedParameter(block_name, card_name, RummyCardToParamValue(card),
                               comment);
      }
    }
    if (full_deck != nullptr) {
      const std::string class_name = full_deck->GetClassName(suit_name);
      if (!class_name.empty()) {
        // Instance name is the last '/'-separated segment of the suit path.
        std::string instance_name = suit_name;
        const auto last_slash = suit_name.find_last_of('/');
        if (last_slash != std::string::npos) {
          instance_name = suit_name.substr(last_slash + 1);
        }
        pin.SetBlockClassMetadata(block_name, class_name, instance_name);
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn bool IsRummyFormat(std::istream &is)
//  \brief Detect whether a stream uses Rummy input format by scanning for markers:
//           - First line is "# use rummy" (case-insensitive)
//           - Non-comment, non-blank content before the first <block> line
//           - Relative suit paths starting with <..
//           - Rummy-specific value syntax: ** power operator, quoted strings,
//             bracket syntax [ ] (vectors/slices), or slice colon inside brackets
bool IsRummyFormat(std::istream &is, const bool command_line) {
  const auto start_pos = is.tellg();
  auto restore_and_return = [&](bool result) {
    is.clear();
    is.seekg(start_pos);
    return result;
  };

  bool first_line = true;
  bool found_block = false;
  std::string line;
  while (std::getline(is, line)) {
    line.erase(std::remove_if(line.begin(), line.end(),
                              [](char c) { return std::isspace(c) && c != ' '; }),
               line.end());
    if (line.empty()) continue;
    auto first_char = line.find_first_not_of(" ");
    if (first_char == std::string::npos) continue;

    // Check first non-blank line for "# use rummy" (case-insensitive)
    if (first_line) {
      first_line = false;
      if (line.compare(first_char, 1, "#") == 0) {
        std::string after_hash = line.substr(first_char + 1);
        auto text_start = after_hash.find_first_not_of(" ");
        if (text_start != std::string::npos) {
          std::string token = after_hash.substr(text_start);
          std::transform(token.begin(), token.end(), token.begin(), ::tolower);
          if (token.compare(0, 10, "use native") == 0) return restore_and_return(false);
          if (token.compare(0, 9, "use rummy") == 0) return restore_and_return(true);
        }
        continue;
      }
    } else {
      if (line.compare(first_char, 1, "#") == 0) continue;
    }

    if (line.compare(first_char, 1, "<") == 0) {
      if (line.size() > first_char + 2 && line.compare(first_char + 1, 2, "..") == 0) {
        return restore_and_return(true);
      }
      found_block = true;
      continue;
    }

    // Non-comment, non-blank content before the first block = Rummy global variable
    // Disable for command line modifications
    if (!command_line && !found_block) {
      return restore_and_return(true);
    }

    // Rummy-specific syntax in the value part
    auto eq_pos = line.find('=');
    if (eq_pos != std::string::npos) {
      std::string name_part = line.substr(first_char, eq_pos - first_char);
      if (name_part.find_first_of(".[") != std::string::npos) {
        return restore_and_return(true);
      }

      std::string value_part = SanitizeString(line.substr(eq_pos + 1));
      // do not include +- because they can be used in exponential notation.
      // / can be used in command line arguments
      if (value_part.find_first_of("*\"[%^|") != std::string::npos) {
        return restore_and_return(true);
      }
    }
    // Slice syntax on the LHS: name[:2] or name[0:2]
    std::string lhs =
        line.substr(first_char, eq_pos == std::string::npos ? std::string::npos
                                                            : eq_pos - first_char);
    if (lhs.find('[') != std::string::npos) {
      return restore_and_return(true);
    }
  }
  return restore_and_return(false);
}

//! \fn bool ParameterInput::IsRummyFormat(const std::string &filename)
//  \brief Detect whether a file uses Rummy input format. Delegates to the stream
//  overload.
bool IsRummyFormat(const std::string &filename) {
  std::ifstream file(filename);
  if (!file.is_open()) return false;
  return IsRummyFormat(file, false);
}

//----------------------------------------------------------------------------------------
//! \fn void ParameterInput::SyncDeckFromStorage()
//  \brief Seed the Rummy Deck from the current param_storage_ contents.
void SyncDeckFromStorage(ParameterInput &pin, Rummy::DeckBase &deck) {
  std::map<std::string, std::map<std::string, Rummy::Card>> new_cards;
  std::vector<std::string> new_suits;
  std::map<std::string, std::vector<std::string>> new_card_map;

  // Register a single card into the three structures, adding the suit on first use.
  auto register_card = [&](const std::string &suit, const std::string &card_name,
                           Rummy::Card card) {
    if (new_cards.find(suit) == new_cards.end()) {
      new_suits.push_back(suit);
      new_card_map[suit] = {};
    }
    new_card_map[suit].push_back(card_name);
    new_cards[suit][card_name] = std::move(card);
  };

  for (const auto &block : pin.GetBlocks()) {
    // Collapse the block name into a Rummy suit: non-empty '/' segments joined by '/'.
    // A block that is only "/" (global scope) maps to suit "/".
    std::string suit = "/";
    {
      std::string assembled;
      std::istringstream bss(block.name);
      std::string part;
      while (std::getline(bss, part, '/')) {
        if (!part.empty()) {
          if (!assembled.empty()) assembled += '/';
          assembled += part;
        }
      }
      if (!assembled.empty()) suit = assembled;
    }

    for (const auto &param : block.params) {
      // Vector variants expand to one card per element: name[0], name[1], ...
      if (std::holds_alternative<std::vector<bool>>(param.value)) {
        const auto &vec = std::get<std::vector<bool>>(param.value);
        for (size_t i = 0; i < vec.size(); ++i) {
          std::string cn = param.name + "[" + std::to_string(i) + "]";
          register_card(suit, cn, Rummy::Card(suit, cn, static_cast<bool>(vec[i]), ""));
        }
      } else if (std::holds_alternative<std::vector<int>>(param.value)) {
        const auto &vec = std::get<std::vector<int>>(param.value);
        for (size_t i = 0; i < vec.size(); ++i) {
          std::string cn = param.name + "[" + std::to_string(i) + "]";
          register_card(suit, cn, Rummy::Card(suit, cn, static_cast<double>(vec[i]), ""));
        }
      } else if (std::holds_alternative<std::vector<Real>>(param.value)) {
        const auto &vec = std::get<std::vector<Real>>(param.value);
        for (size_t i = 0; i < vec.size(); ++i) {
          std::string cn = param.name + "[" + std::to_string(i) + "]";
          register_card(suit, cn, Rummy::Card(suit, cn, static_cast<double>(vec[i]), ""));
        }
      } else if (std::holds_alternative<std::vector<std::string>>(param.value)) {
        const auto &vec = std::get<std::vector<std::string>>(param.value);
        for (size_t i = 0; i < vec.size(); ++i) {
          std::string cn = param.name + "[" + std::to_string(i) + "]";
          register_card(suit, cn, Rummy::Card(suit, cn, vec[i], ""));
        }
      } else {
        register_card(suit, param.name,
                      ParamValueToRummyCard(suit, param.name, param.value));
      }
    }
  }

  deck.SeedGlobals(new_cards, new_suits, new_card_map);
}

} // namespace parthenon
