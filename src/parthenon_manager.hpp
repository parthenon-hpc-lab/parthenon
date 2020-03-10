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

#ifndef PARTHENON_MANAGER_HPP
#define PARTHENON_MANAGER_HPP

#include <memory>

#include "argument_parser.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "outputs/outputs.hpp"
#include "interface/PropertiesInterface.hpp"
#include "interface/StateDescriptor.hpp"

namespace parthenon {

enum class ParthenonStatus {ok, complete, error};

class ParthenonManager {
  public:
    ParthenonManager() = default;
    ParthenonStatus ParthenonInit(int argc, char *argv[]);
    ParthenonStatus ParthenonFinalize();

    bool Restart() { return (arg.restart_filename == nullptr ? false : true); }
    void ProcessProperties(std::unique_ptr<ParameterInput>& pin, Properties_t& properties);
    void ProcessPackages(std::unique_ptr<ParameterInput>& pin, Packages_t& packages);

    // member data
    std::unique_ptr<ParameterInput> pinput;
    std::unique_ptr<Mesh> pmesh;
    std::unique_ptr<Outputs> pouts;
  private:
    ArgParse arg;
};







}

#endif
