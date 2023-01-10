#ifndef RENDER_ASCENT_HPP_
#define RENDER_ASCENT_HPP_

#include <parthenon/package.hpp>

#include "advection_driver.hpp"
#include "advection_package.hpp"
#include "config.hpp"
#include "defs.hpp"
#include "interface/variable_pack.hpp"
#include "utils/error_checking.hpp"

#include "ascent.hpp"
#include "conduit_blueprint.hpp"


void render_ascent(parthenon::Mesh *mesh, parthenon::ParameterInput *pin,
                   parthenon::SimTime &tm);

#endif // RENDER_ASCENT_HPP_
