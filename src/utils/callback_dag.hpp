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
#ifndef UTILS_CALLBACK_DAG_HPP_
#define UTILS_CALLBACK_DAG_HPP_
// This file was made in part with generative AI.

#include <map>
#include <ostream>
#include <set>
#include <string>
#include <vector>

namespace parthenon {

/// Directed Acyclic Graph (DAG) for managing callback execution order.
///
/// This class builds a dependency graph from callback specifications and computes
/// a valid execution order via topological sorting. It detects and reports cycles
/// with detailed path information.
class CallbackDAG {
 public:
  CallbackDAG() = default;

  /// Add a node to the graph.
  /// @param name The unique identifier for this node (typically unit name)
  void AddNode(const std::string &name);

  /// Add a directed edge from 'from' to 'to'.
  /// This means 'from' must execute before 'to'.
  /// @param from The source node (executes first)
  /// @param to The destination node (executes after)
  void AddEdge(const std::string &from, const std::string &to);

  /// Compute topological sort of the graph.
  /// @return Vector of node names in valid execution order
  /// @throws std::runtime_error if graph contains a cycle
  std::vector<std::string> TopologicalSort() const;

  /// Write the graph in GraphViz DOT format for visualization.
  /// @param stream Output stream to write to
  void WriteGraphViz(std::ostream &stream) const;

  /// Stream insertion operator for easy output.
  friend std::ostream &operator<<(std::ostream &stream, const CallbackDAG &dag);

 private:
  /// Detect cycles in the graph using DFS.
  /// @throws std::runtime_error with cycle path if a cycle is detected
  void ValidateAcyclic() const;

  /// DFS helper for topological sort.
  /// @param node Current node being visited
  /// @param visited Set of permanently visited nodes
  /// @param rec_stack Set of nodes in current recursion stack (for cycle detection)
  /// @param result Output vector for topological order (reverse post-order)
  void TopologicalSortDFS(const std::string &node, std::set<std::string> &visited,
                          std::set<std::string> &rec_stack,
                          std::vector<std::string> &result) const;

  /// DFS helper for cycle detection with path tracking.
  /// @param node Current node being visited
  /// @param visited Set of permanently visited nodes
  /// @param rec_stack Set of nodes in current recursion stack
  /// @param path Current path from root to node
  /// @return true if cycle detected, false otherwise
  bool DetectCycleDFS(const std::string &node, std::set<std::string> &visited,
                      std::set<std::string> &rec_stack,
                      std::vector<std::string> &path) const;

  std::set<std::string> nodes_;
  std::map<std::string, std::vector<std::string>> adjacency_list_;
};

} // namespace parthenon

#endif // UTILS_CALLBACK_DAG_HPP_
