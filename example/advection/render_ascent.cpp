#include "render_ascent.hpp"
#include "defs.hpp"
#include "interface/metadata.hpp"
#include "interface/variable.hpp"
#include "mesh/domain.hpp"
#include <memory>
#include <string>

using namespace parthenon::package::prelude;
using namespace parthenon;

using namespace ascent;
using namespace conduit;

// reference:
// https://llnl-conduit.readthedocs.io/en/latest/blueprint_mesh.html#complete-uniform-example
//
void render_ascent(Mesh *par_mesh, ParameterInput *pin, SimTime const &tm) {
  // Ascent needs the MPI communicator we are using
  Ascent a;
  Node ascent_opts;
  ascent_opts["mpi_comm"] = MPI_Comm_c2f(MPI_COMM_WORLD);
  a.open(ascent_opts);

  // create root node for the whole mesh
  Node root;

  for (auto &thisMeshBlock : par_mesh->block_list) {
    // create a unique id for this MeshBlock
    const std::string &meshblock_name = "domain_" + std::to_string(thisMeshBlock->gid);
    Node &mesh = root[meshblock_name];

    // add basic state info
    mesh["state/domain_id"] = thisMeshBlock->gid;
    mesh["state/cycle"] = tm.ncycle;
    mesh["state/time"] = tm.time;

    auto &bounds = thisMeshBlock->cellbounds;
    auto ib = bounds.GetBoundsI(IndexDomain::entire);
    auto jb = bounds.GetBoundsJ(IndexDomain::entire);
    auto kb = bounds.GetBoundsK(IndexDomain::entire);
    int nx = ib.e - ib.s + 1;
    int ny = jb.e - jb.s + 1;
    int nz = kb.e - kb.s + 1;
    uint64_t ncells = nx * ny * nz;

    auto ib_int = bounds.GetBoundsI(IndexDomain::interior);
    auto jb_int = bounds.GetBoundsJ(IndexDomain::interior);
    auto kb_int = bounds.GetBoundsK(IndexDomain::interior);

    auto &coords = thisMeshBlock->coords;
    Real dx1 = coords.CellWidth<X1DIR>(ib.s, jb.s, kb.s);
    Real dx2 = coords.CellWidth<X2DIR>(ib.s, jb.s, kb.s);
    Real dx3 = coords.CellWidth<X3DIR>(ib.s, jb.s, kb.s);
    std::array<Real, 3> corner = coords.GetXmin();

    // create the coordinate set
    mesh["coordsets/coords/type"] = "uniform";

    mesh["coordsets/coords/dims/i"] = nx + 1;
    mesh["coordsets/coords/dims/j"] = ny + 1;
    if (nz > 1) {
      mesh["coordsets/coords/dims/k"] = nz + 1;
    }

    // add origin and spacing to the coordset (optional)
    mesh["coordsets/coords/origin/x"] = corner[0];
    mesh["coordsets/coords/origin/y"] = corner[1];
    if (nz > 1) {
      mesh["coordsets/coords/origin/z"] = corner[2];
    }

    mesh["coordsets/coords/spacing/dx"] = dx1;
    mesh["coordsets/coords/spacing/dy"] = dx2;
    if (nz > 1) {
      mesh["coordsets/coords/spacing/dz"] = dx3;
    }

    // add the topology
    mesh["topologies/topo/type"] = "uniform";
    mesh["topologies/topo/coordset"] = "coords";

    // indicate ghost zones with ascent_ghosts set to 1
    Node &n_field = mesh["fields/ascent_ghosts"];
    n_field["association"] = "element";
    n_field["topology"] = "topo";
    n_field["values"].set(DataType::int32(ncells));
    int32_array vals_array = n_field["values"].value();

    int idx = 0;
    for (int k = kb.s; k <= kb.e; k++) {
      for (int j = jb.s; j <= jb.e; j++) {
        for (int i = ib.s; i <= ib.e; i++) {
          if ((i < ib_int.s) || (ib_int.e < i) || (j < jb_int.s) || (jb_int.e < j) ||
              ((nz > 1) && ((k < kb_int.s) || (kb_int.e < k)))) {
            vals_array[idx] = 1;
          }
          idx++;
        }
      }
    }

    // create a field for each component of each variable pack
    auto &mbd = thisMeshBlock->meshblock_data.Get();

    for (auto &vars : mbd->GetCellVariableVector()) {
      const std::string packname = vars->label();
      auto const &labels = vars->metadata().getComponentLabels();
      auto const &data = vars->data;

      for (int icomp = 0; icomp < labels.size(); ++icomp) {
        const std::string varname = packname + ":" + labels.at(icomp);
        mesh["fields/" + varname + "/association"] = "element";
        mesh["fields/" + varname + "/topology"] = "topo";
        mesh["fields/" + varname + "/values"].set_external(&data(icomp, 0, 0, 0), ncells);
      }
    }
  }

  // make sure we conform:
  Node verify_info;
  if (!blueprint::mesh::verify(root, verify_info)) {
    if (parthenon::Globals::my_rank == 0) {
      std::cout << "blueprint::mesh::verify failed!" << std::endl;
    }
    verify_info.print();
  }
  a.publish(root);

  // setup actions
  Node actions;
  Node &add_act = actions.append();
  add_act["action"] = "add_scenes";

  // declare a scene (s1) with one plot (p1)
  Node &scenes = add_act["scenes"];
  scenes["s1/plots/p1/type"] = "pseudocolor";
  scenes["s1/plots/p1/field"] = "advected:Advected_0_0";

  // Set the output file name (ascent will add ".png")
  scenes["s1/image_prefix"] = "ascent_render";

  // execute the actions
  a.execute(actions);

  // close ascent
  a.close();
}
