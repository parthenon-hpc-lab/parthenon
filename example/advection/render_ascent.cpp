#include "render_ascent.hpp"
#include "defs.hpp"
#include "mesh/domain.hpp"

using namespace parthenon::package::prelude;
using namespace parthenon;

using namespace ascent;
using namespace conduit;

void render_ascent(Mesh *par_mesh, ParameterInput *pin, SimTime const &tm) {
  //
  // reference:
  // https://llnl-conduit.readthedocs.io/en/latest/blueprint_mesh.html#complete-uniform-example

  // call Ascent every ascent_interval timesteps
  const int ascent_interval = 10;
  static int counter = 0;
  if (!(counter % ascent_interval == 0)) {
    return;
  }
  std::cout << "\nRendering ascent (step = " << counter << ")..." << std::endl;
  counter++;

  // Ascent needs the MPI communicator we are using
  Ascent a;
  Node ascent_opts;
  ascent_opts["mpi_comm"] = MPI_Comm_c2f(MPI_COMM_WORLD);
  a.open(ascent_opts);

  Node root;

  for (auto &thisMeshBlock : par_mesh->block_list) {
    // create a unique id for this MeshBlock
    const std::string &meshblock_name = "domain_" + std::to_string(thisMeshBlock->gid);
    Node &mesh = root[meshblock_name];

    // add basic state info
    mesh["state/domain_id"] = thisMeshBlock->gid;
    mesh["state/cycle"] = tm.ncycle;
    mesh["state/time"] = tm.time;

    std::cout << "creating mesh for MeshBlock " << meshblock_name << std::endl;

    auto &bounds = thisMeshBlock->cellbounds;
    auto ib = bounds.GetBoundsI(IndexDomain::entire);
    auto jb = bounds.GetBoundsJ(IndexDomain::entire);
    auto kb = bounds.GetBoundsK(IndexDomain::entire);
    int nx = ib.e - ib.s + 1;
    int ny = jb.e - jb.s + 1;
    int nz = kb.e - kb.s + 1;
    uint64_t ncells = nx * ny * nz;

    auto &coords = thisMeshBlock->coords;
    Real dx1 = coords.CellWidth<X1DIR>(ib.s, jb.s, kb.s);
    Real dx2 = coords.CellWidth<X2DIR>(ib.s, jb.s, kb.s);
    Real dx3 = coords.CellWidth<X3DIR>(ib.s, jb.s, kb.s);
    std::array<Real, 3> corner = coords.GetXmin();

    auto &mbd = thisMeshBlock->meshblock_data.Get();
    auto &vars = mbd->PackVariables();

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
    // this case is simple b/c it's implicitly derived from the coordinate set
    mesh["topologies/topo/type"] = "uniform";
    // reference the coordinate set by name
    mesh["topologies/topo/coordset"] = "coords";

    // for each variable:

    // add a simple element-associated field
    mesh["fields/my_var_name/association"] = "element";
    // reference the topology this field is defined on by name
    mesh["fields/my_var_name/topology"] = "topo";

    // set the field values
    int nvar = 0;
    mesh["fields/my_var_name/values"].set_external(&vars(nvar, 0, 0, 0), ncells);
  }

  // make sure we conform:
  Node verify_info;
  if (!blueprint::mesh::verify(root, verify_info)) {
    std::cout << "Verify failed!" << std::endl;
    verify_info.print();
  }

  // save our mesh to a file that can be read by VisIt
  // this will create the file: complete_uniform_mesh_example.root
  // which includes the mesh blueprint index and the mesh data
  // conduit::relay::io::blueprint::save_mesh(root, "complete_mesh", "json");

  a.publish(root);

  // setup actions
  Node actions;
  Node &add_act = actions.append();
  add_act["action"] = "add_scenes";

  // declare a scene (s1) with one plot (p1)
  // to render the dataset
  Node &scenes = add_act["scenes"];
  scenes["s1/plots/p1/type"] = "pseudocolor";
  scenes["s1/plots/p1/field"] = "my_var_name";

  // Set the output file name (ascent will add ".png")
  scenes["s1/image_prefix"] = "ascent_render";

  // execute the actions
  a.execute(actions);

  // close ascent
  a.close();
}
