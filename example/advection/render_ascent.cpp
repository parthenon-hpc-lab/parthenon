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
  counter++;
  if (!(counter % ascent_interval == 0)) {
    return;
  }

  std::cout << "\nRendering ascent (step = " << counter << ")..." << std::endl;

  // Ascent needs the MPI communicator we are using
  Ascent a;
  Node ascent_opts;
  ascent_opts["mpi_comm"] = MPI_Comm_c2f(MPI_COMM_WORLD);
  a.open(ascent_opts);

  // vector of all meshes
  std::vector<Node> mesh_vec;

  for (auto &thisMeshBlock : par_mesh->block_list) {
    std::cout << "creating mesh for MeshBlock " << thisMeshBlock << std::endl;

    auto &blockSize = thisMeshBlock->block_size;
    int nx = blockSize.nx1;
    int ny = blockSize.nx2;
    int nz = blockSize.nx3;

    auto &bounds = thisMeshBlock->cellbounds;
    auto ib = bounds.GetBoundsI(IndexDomain::entire);
    auto jb = bounds.GetBoundsJ(IndexDomain::entire);
    auto kb = bounds.GetBoundsK(IndexDomain::entire);

    auto &coords = thisMeshBlock->coords;
    Real dx1 = coords.CellWidth<X1DIR>(ib.s, jb.s, kb.s);
    Real dx2 = coords.CellWidth<X2DIR>(ib.s, jb.s, kb.s);
    Real dx3 = coords.CellWidth<X3DIR>(ib.s, jb.s, kb.s);
    std::array<Real, 3> corner = coords.GetXmin();

    auto &mbd = thisMeshBlock->meshblock_data.Get();
    auto &vars = mbd->PackVariables();

    // create a Conduit node to hold our mesh data
    // (we need to save this in a list for rendering)
    Node mesh;

    // create the coordinate set
    mesh["coordsets/coords/type"] = "uniform";
    mesh["coordsets/coords/dims/i"] = nx;
    mesh["coordsets/coords/dims/j"] = ny;
    mesh["coordsets/coords/dims/k"] = nz;
    uint64_t ncells = nx * ny * nz;

    // add origin and spacing to the coordset (optional)
    mesh["coordsets/coords/origin/x"] = corner[0];
    mesh["coordsets/coords/origin/y"] = corner[1];
    mesh["coordsets/coords/origin/z"] = corner[2];
    mesh["coordsets/coords/spacing/dx"] = dx1;
    mesh["coordsets/coords/spacing/dy"] = dx2;
    mesh["coordsets/coords/spacing/dz"] = dx3;

    // add the topology
    // this case is simple b/c it's implicitly derived from the coordinate set
    mesh["topologies/topo/type"] = "uniform";
    // reference the coordinate set by name
    mesh["topologies/topo/coordset"] = "coords";

    // for each variable:

    // add a simple element-associated field
    mesh["fields/ele_example/association"] = "element";
    // reference the topology this field is defined on by name
    mesh["fields/ele_example/topology"] = "topo";
    
    // set the field values
    int nvar = 0;
    mesh["fields/ele_example/values"].set_external(&vars(nvar, 0, 0, 0), ncells);

    // make sure we conform:
    Node verify_info;
    if (!blueprint::mesh::verify(mesh, verify_info)) {
      std::cout << "Verify failed!" << std::endl;
      verify_info.print();
    }

    // save our mesh to a file that can be read by VisIt
    // this will create the file: complete_uniform_mesh_example.root
    // which includes the mesh blueprint index and the mesh data
    conduit::relay::io::blueprint::save_mesh(mesh, "complete_uniform_mesh_example",
                                             "json");

    mesh_vec.emplace_back(mesh);
  }

  for (auto &mesh : mesh_vec) {
    // publish mesh to ascent
    a.publish(mesh);
  }

  // setup actions
  Node actions;
  Node &add_act = actions.append();
  add_act["action"] = "add_scenes";

  // declare a scene (s1) with one plot (p1)
  // to render the dataset
  Node &scenes = add_act["scenes"];
  scenes["s1/plots/p1/type"] = "pseudocolor";
  scenes["s1/plots/p1/field"] = "ele_example";

  // Set the output file name (ascent will add ".png")
  scenes["s1/image_prefix"] = "out_ascent_render_uniform";

  // execute the actions
  a.execute(actions);

  // close ascent
  a.close();
}
