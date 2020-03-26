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
#include "Params.hpp"

using namespace parthenon;

int main() {
  Params p;
  p.Add("di",std::vector<int>{1,2,3});
  p.Add("i",4);
  p.Add("f",16.7f);
  p.Add("f",5537.8f); // replace f with an integer!!!
  p.Add("dV",std::vector<double>{1.,2.,3.});
  p.Add("str",std::string("hello dolly"));
  p.Add("strV",std::vector<std::string>({std::string("hello"),std::string("dolly")}));
  p.list();

  std::string key;
  /*
  key="i"; std::cout << key << ":" << p.Get<int>(key) << std::endl;
  key="f"; std::cout << key << ":" << p.Get<float>(key) << std::endl;
  key="str"; std::cout << key << ":" << p.Get<std::string>(key) << std::endl;
  */

  int j;
  p.Get("i",j);
  auto k = p.GetInt(std::string("i"));
  auto l = p.Get<int>("i");
  std::cout << "Got l = "<<l<<std::endl;
  //  float fk = p.GetFloat(std::string("i"));
  std::cout << "GOT k="<<k<<std::endl;
  //  std::cout << "GOT fk="<<fk<<std::endl;

  p.reset();

  // die horribly
  //  key="i"; std::cout << key << ":" << p.Get<int>(key) << std::endl;

  // destructor automatically called for p
  return 0;
}

