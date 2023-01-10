#include "render_ascent.hpp"

using namespace parthenon::package::prelude;
using namespace parthenon;

using namespace ascent;
using namespace conduit;

void render_ascent(Mesh *par_mesh, ParameterInput *pin, SimTime const &tm) {
  //
  // reference:
  // https://llnl-conduit.readthedocs.io/en/latest/blueprint_mesh.html#complete-uniform-example
  //

  static int counter = 0;
  const int ascent_interval = 10;
  counter++;
  if (!(counter % ascent_interval == 0)) {
    return;
  }
  std::cout << "rendering ascent (counter = " << counter << ")...\n" << std::flush;

  // create a Conduit node to hold our mesh data
  Node mesh;

  // create the coordinate set
  mesh["coordsets/coords/type"] = "uniform";
  mesh["coordsets/coords/dims/i"] = 3;
  mesh["coordsets/coords/dims/j"] = 3;
  // add origin and spacing to the coordset (optional)
  mesh["coordsets/coords/origin/x"] = -10.0;
  mesh["coordsets/coords/origin/y"] = -10.0;
  mesh["coordsets/coords/spacing/dx"] = 10.0;
  mesh["coordsets/coords/spacing/dy"] = 10.0;

  // add the topology
  // this case is simple b/c it's implicitly derived from the coordinate set
  mesh["topologies/topo/type"] = "uniform";
  // reference the coordinate set by name
  mesh["topologies/topo/coordset"] = "coords";

  // add a simple element-associated field
  mesh["fields/ele_example/association"] = "element";
  // reference the topology this field is defined on by name
  mesh["fields/ele_example/topology"] = "topo";
  // set the field values, for this case we have 4 elements
  mesh["fields/ele_example/values"].set(DataType::float64(4));

  float64 *ele_vals_ptr = mesh["fields/ele_example/values"].value();

  for (int i = 0; i < 4; i++) {
    ele_vals_ptr[i] = float64(i);
  }

  // make sure we conform:
  Node verify_info;
  if (!blueprint::mesh::verify(mesh, verify_info)) {
    std::cout << "Verify failed!" << std::endl;
    verify_info.print();
  }

  // save our mesh to a file that can be read by VisIt
  //
  // this will create the file: complete_uniform_mesh_example.root
  // which includes the mesh blueprint index and the mesh data
  conduit::relay::io::blueprint::save_mesh(mesh, "complete_uniform_mesh_example", "json");

  // now render with Ascent
  Ascent a;

  // we use the mpi handle provided by the fortran interface
  // since it is simply an integer
  Node ascent_opts;
  ascent_opts["mpi_comm"] = MPI_Comm_c2f(MPI_COMM_WORLD);
  a.open(ascent_opts);

  // publish mesh to ascent
  // IMPORTANT: an ascent 'mesh' object must be created/published for *each* Parthenon
  // meshblock
  a.publish(mesh);

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
