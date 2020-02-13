//========================================================================================
// (C) (or copyright) 2020. Triad National Security, LLC. All rights reserved.
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
#ifndef STATEDICT_H_PK
#define STATEDICT_H_PK

#include <map>
#include <set>
#include <string>
#include <iostream>
#include "interface/Metadata.hpp"

class StateDictionary : public std::map<std::string, Metadata> {
  public:
    void AddState(const std::string& field_name, Metadata& m) {
#if 0
      std::cerr << "Adding " << field_name << std::endl;
      if ((!m.isSet(m.materials)) && this->count(field_name)) {
        std::cerr << "not adding " << field_name << " to state: " << m.isSet(m.materials) << " " << this->count(field_name) << std::endl;
	// already added to state
        // make sure it's metadata matches what's already in there
        // DOES THIS WORK?
        if (m != this[field_name]) {
          // THROW AN ERROR FOR NOW
        }
      } else {
        std::cerr << "inserting..." << std::endl;
#endif
        this->insert( std::pair<std::string, Metadata>(field_name, m) );
        //_fields.insert(field_name);
        //_fields_metadata.insert( );
//      }
    }
  //std::set<std::string> GetFields() {return _fields;}
  //Metadata GetMetadata(std::string& field_name) {return _fields_metadata[field_name];}

  private:
  //std::set<std::string> _fields;
  //std::map<std::string, Metadata> _fields_metadata;
};
#endif
