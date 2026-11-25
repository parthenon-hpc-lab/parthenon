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
#include "callbacks.hpp"

#include <algorithm>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace parthenon {

void CallbackDAG::AddNode(const std::string &name) { nodes_.insert(name); }

void CallbackDAG::AddEdge(const std::string &from, const std::string &to) {
  nodes_.insert(from);
  nodes_.insert(to);

  adjacency_list_[from].push_back(to);
}

std::vector<std::string> CallbackDAG::TopologicalSort() const {
  ValidateAcyclic();

  std::set<std::string> visited;
  std::set<std::string> rec_stack;
  std::vector<std::string> result;

  for (const auto &node : nodes_) {
    if (visited.find(node) == visited.end()) {
      TopologicalSortDFS(node, visited, rec_stack, result);
    }
  }

  // DFS produces reverse topological order, so reverse it
  std::reverse(result.begin(), result.end());
  return result;
}

void CallbackDAG::TopologicalSortDFS(const std::string &node,
                                     std::set<std::string> &visited,
                                     std::set<std::string> &rec_stack,
                                     std::vector<std::string> &result) const {
  visited.insert(node);
  rec_stack.insert(node);

  auto it = adjacency_list_.find(node);
  if (it != adjacency_list_.end()) {
    for (const auto &neighbor : it->second) {
      if (visited.find(neighbor) == visited.end()) {
        TopologicalSortDFS(neighbor, visited, rec_stack, result);
      }
    }
  }

  rec_stack.erase(node);
  result.push_back(node);
}

void CallbackDAG::ValidateAcyclic() const {
  std::set<std::string> visited;
  std::set<std::string> rec_stack;
  std::vector<std::string> path;

  for (const auto &node : nodes_) {
    if (visited.find(node) == visited.end()) {
      if (DetectCycleDFS(node, visited, rec_stack, path)) {
        // Cycle detected - path contains the cycle
        std::ostringstream oss;
        oss << "Cyclic dependency detected: ";
        for (size_t i = 0; i < path.size(); ++i) {
          if (i > 0) oss << " -> ";
          oss << path[i];
        }
        throw std::runtime_error(oss.str());
      }
    }
  }
}

bool CallbackDAG::DetectCycleDFS(const std::string &node, std::set<std::string> &visited,
                                 std::set<std::string> &rec_stack,
                                 std::vector<std::string> &path) const {
  visited.insert(node);
  rec_stack.insert(node);
  path.push_back(node);

  auto it = adjacency_list_.find(node);
  if (it != adjacency_list_.end()) {
    for (const auto &neighbor : it->second) {
      // If neighbor is in recursion stack, we found a cycle
      if (rec_stack.find(neighbor) != rec_stack.end()) {
        // Add the neighbor to complete the cycle path
        path.push_back(neighbor);
        return true;
      }

      // Recurse on unvisited neighbors
      if (visited.find(neighbor) == visited.end()) {
        if (DetectCycleDFS(neighbor, visited, rec_stack, path)) {
          return true;
        }
      }
    }
  }

  rec_stack.erase(node);
  path.pop_back();
  return false;
}

void CallbackDAG::WriteGraphViz(std::ostream &stream) const {
  stream << "digraph {\n";
  stream << "  node [fontname=\"Helvetica,Arial,sans-serif\"]\n";
  stream << "  edge [fontname=\"Helvetica,Arial,sans-serif\"]\n";

  // Write all nodes
  for (const auto &node : nodes_) {
    stream << "  \"" << node << "\";\n";
  }

  // Write all edges
  for (const auto &[from, neighbors] : adjacency_list_) {
    for (const auto &to : neighbors) {
      stream << "  \"" << from << "\" -> \"" << to << "\";\n";
    }
  }

  stream << "}\n";
}

std::ostream &operator<<(std::ostream &stream, const CallbackDAG &dag) {
  dag.WriteGraphViz(stream);
  return stream;
}

} // namespace parthenon
