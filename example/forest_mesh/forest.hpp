//========================================================================================
// (C) (or copyright) 2023. Triad National Security, LLC. All rights reserved.
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
#ifndef FOREST_HPP_
#define FOREST_HPP_

#include <array>
#include <map>
#include <set>
#include <vector>
#include <memory>
#include <tuple> 
#include <unordered_map>
#include <unordered_set> 

#include "basic_types.hpp"
#include "defs.hpp"
#include "mesh/logical_location.hpp"
#include "utils/bit_hacks.hpp"
#include "utils/indexer.hpp"

namespace parthenon {
namespace forest { 
  enum class Direction : uint {I = 0, J = 1, K = 2};

  struct RelativeOrientation { 
    RelativeOrientation() : dir_connection{0, 1, 2}, dir_flip{false, false, false} {};
  
    void SetDirection(Direction origin, Direction neighbor, bool reversed = false) { 
      dir_connection[static_cast<uint>(origin)] = static_cast<uint>(neighbor);
      dir_flip[static_cast<uint>(origin)] = reversed;
    }
  
    LogicalLocation Transform(const LogicalLocation &loc_in) const;
  
    int dir_connection[3]; 
    bool dir_flip[3];
  };

   using ForestLocation = std::pair<std::uint64_t, LogicalLocation>; 

  // We don't allow for periodic boundaries, since we can encode periodicity through connectivity in the forest
  class Tree { 
   public: 
    Tree(int ndim, int root_level, RegionSize domain = RegionSize());

    template <class... Ts> 
    static std::shared_ptr<Tree> create(Ts&&... args) {
      return std::make_shared<Tree>(std::forward<Ts>(args)...);
    }

    // Methods for modifying the tree  
    int Refine(LogicalLocation ref_loc);
    int Derefine(LogicalLocation ref_loc);

    // Methods for getting block properties 
    std::vector<ForestLocation> GetMeshBlockList() const;
    RegionSize GetBlockDomain(LogicalLocation loc) const;    

    // Methods for building tree connectivity
    void AddNeighbor(int location_idx, std::shared_ptr<Tree> neighbor_tree, RelativeOrientation orient) { 
      neighbors[location_idx].push_back(std::make_pair(neighbor_tree, orient));   
    }
    void SetId(std::uint64_t id) {my_id = id;}

    // TODO: Remove this function, only here for testing
    void Print(std::string fname) const {
      FILE * pFile;
      pFile = fopen(fname.c_str(), "w");
      for (const auto &l : leaves) 
        fprintf(pFile, "%i, %i, %i\n", l.level(), l.lx1(), l.lx2());
      fclose(pFile);
    }
    const std::unordered_set<LogicalLocation>& GetLeaves() const { return leaves;}
    
   private:
    int ndim;
    std::uint64_t my_id;  
    std::unordered_set<LogicalLocation> leaves; 
    std::unordered_set<LogicalLocation> internal_nodes; 
    std::array<std::vector<std::pair<std::shared_ptr<Tree>, RelativeOrientation>>, 27> neighbors;
    RegionSize domain; 
  };
  
  class Forest { 
   public: 
    std::vector<std::shared_ptr<Tree>> trees;

    std::vector<ForestLocation> GetMeshBlockList() const;
    RegionSize GetBlockDomain(ForestLocation loc) const {
      return trees[loc.first]->GetBlockDomain(loc.second);
    }

    static Forest AthenaXX(RegionSize mesh_size, RegionSize block_size, std::array<bool, 3> periodic);
  }; 

} // namespace forest
} // namespace parthenon

#endif // FOREST_HPP_
